import kaggle
import polars as pl
import shutil
from functools import lru_cache
from periodic_table import PeriodicTable
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pickle
from multiprocessing import Pool
from symbolic_nn_tests.dataloader import DATASET_DIR
import warnings
from tqdm.auto import tqdm
from loguru import logger


warnings.filterwarnings(action="ignore", category=UserWarning)


PUBCHEM_DIR = DATASET_DIR / "PubChem"

ORBITALS = {
    "s": (0, 2),
    "p": (1, 6),
    "d": (2, 10),
    "f": (3, 14),
}


def collate(batch):
    x0_in, x1_in, y_in = list(zip(*batch))
    x0_out = torch.nested.as_nested_tensor(list(x0_in))
    x1_out = torch.nested.as_nested_tensor(list(x1_in))
    y_out = torch.as_tensor(y_in)
    return (x0_out, x1_out), y_out


def pubchem(*args, **kwargs):
    PUBCHEM_DIR.mkdir(exist_ok=True, parents=True)
    return get_dataset()


def get_dataset():
    if not (
        "pubchem_x0.pickle"
        and "pubchem_x1.pickle"
        and "pubchem_y.pickle" in (x.name for x in PUBCHEM_DIR.iterdir())
    ):
        construct_dataset("pubchem")
    else:
        logger.info("Pre-existing dataset detected!")
    logger.info("Dataset loaded!")
    return TensorDataset(*load_dataset("pubchem"))


def construct_dataset(filename):
    logger.info("Constructing dataset...")
    df = construct_ds_dataframe(filename)
    save_dataframe_to_dataset(df, PUBCHEM_DIR / f"{filename}.pickle")
    logger.info("Dataset constructed!")


def construct_ds_dataframe(filename):
    logger.info("Constructing dataset dataframe...")
    df = add_molecule_encodings(construct_raw_dataset(filename))
    # NOTE: This kind of checkpointing will be used throughout the construction process It doesn't
    # take much disk space, it lets the GC collect out-of-scope data from the construction process
    # and it makes it easier to debug if construction fails
    parquet_file = PUBCHEM_DIR / f"{filename}.parquet"
    df.write_parquet(parquet_file)
    logger.info("Dataset dataframe constructed!")
    return pl.read_parquet(parquet_file)


def construct_raw_dataset(filename):
    logger.info("Constructing raw dataset...")
    df = collate_dataset()
    parquet_file = PUBCHEM_DIR / f"{filename}_raw.parquet"
    df.write_parquet(parquet_file)
    logger.info("Raw dataset constructed!")
    return pl.read_parquet(parquet_file)


def collate_dataset():
    logger.info("Collating dataset...")
    if not (PUBCHEM_DIR.exists() and len(tuple(PUBCHEM_DIR.glob("*.json")))):
        fetch_dataset()

    df = pl.concat(
        map(pl.read_json, PUBCHEM_DIR.glob("*.json")),
    ).drop("id")
    logger.info("dataset collated!")
    return df


def fetch_dataset():
    logger.info("Fetching dataset...")
    kaggle.api.dataset_download_files(
        "burakhmmtgl/predict-molecular-properties", quiet=False, path=DATASET_DIR
    )
    shutil.unpack_archive(DATASET_DIR / "predict-molecular-properties.zip", PUBCHEM_DIR)
    logger.info("Dataset fetched!")


@lru_cache(maxsize=1)
def get_periodic_table():
    return PeriodicTable()


def add_molecule_encodings(df):
    atom_properties, atom_electrons = encode_molecules(df["atoms"])
    return df.with_columns(
        atom_properties=atom_properties, atom_electrons=atom_electrons
    )


def encode_molecules(series):
    # Yes, it is gross and RAM inefficient to do it this way but i dont have all day...
    with Pool() as p:
        molecules = p.map(encode_molecule, series)
    properties, electrons = zip(*molecules)
    return pl.Series(properties), pl.Series(electrons)


def encode_molecule(molecule):
    properties, electrons = zip(*_encode_molecule(molecule))
    properties = pl.Series(properties)
    return properties, electrons


def _encode_molecule(molecule):
    for atom in molecule:
        properties, electrons = encode_atom(atom["type"])
        yield np.array([*properties, *atom["xyz"]]), pl.Series(electrons)


def encode_atom(atom):
    element = get_periodic_table().search_symbol(atom)
    return (
        np.array(
            [
                # n and z need to be scaled somehow to normalize to approximately 1
                # Because this is kind arbitrary i've decided to scale relative to
                # Fermium (n = 100)
                element.atomic / 100.0,
                element.atomic_mass / 257.0,
                element.electron_affinity / 350.0,  # Highest known is just below 350
                element.electronegativity_pauling
                / 4.0,  # Max theoretical val is 4.0 here
            ],
        ),
        encode_electron_config(element.electron_configuration),
    )


def encode_electron_config(config):
    return np.array([encode_orbital(x) for x in config.split()])


def encode_orbital(orbital):
    shell, subshell, *n = orbital
    shell = int(shell)
    n = int("".join(n))
    azimuthal, capacity = ORBITALS[subshell]
    return np.array(
        [
            1.0
            / shell,  # This is the simplest way to normalize shell, as shells become less distinct as n increases
            azimuthal / 4.0,  # This is simply normalizing the azimuthal quantum number
            n / capacity,  # Basically encoding this as a proportion of "fullness"
        ],
    )


def save_dataframe_to_dataset(df, filename):
    logger.info("Saving dataset to tensors...")
    with (filename.parent / f"{filename.stem}_x0{filename.suffix}").open("wb") as f:
        pickle.dump(properties_to_tensor(df).float(), f)
    with (filename.parent / f"{filename.stem}_x1{filename.suffix}").open("wb") as f:
        pickle.dump(electrons_to_tensor(df).float(), f)
    with (filename.parent / f"{filename.stem}_y{filename.suffix}").open("wb") as f:
        pickle.dump(df["En"].to_torch().float(), f)
    del df
    logger.info("Tensors saved!")


def chunked_df(df, n):
    chunk_size = (len(df) // n) + 1
    chunk_boundaries = [*range(0, len(df), chunk_size), len(df)]
    chunk_ranges = list(zip(chunk_boundaries[:-1], chunk_boundaries[1:]))
    yield from (df[i:j] for i, j in chunk_ranges)


def properties_to_tensor(df):
    with Pool() as p:
        out = torch.cat(
            p.map(
                property_chunk_to_torch, chunked_df(df["atom_properties"], p._processes)
            )
        )
    return out


def property_chunk_to_torch(chunk):
    return torch.nested.nested_tensor([properties_to_torch(x) for x in chunk])


def properties_to_torch(p):
    return torch.stack(tuple(map(pl.Series.to_torch, p)))


def electrons_to_tensor(df):
    return torch.nested.nested_tensor(
        [
            molecule_electrons_to_torch(e)
            for e in tqdm(df["atom_electrons"], desc="Converting molecules to orbitals")
        ]
    )


def molecule_electrons_to_torch(e):
    return torch.stack([atom_electrons_to_torch(x) for x in e])


def atom_electrons_to_torch(e):
    # pytorch doesn't like doubly nested tensors, and the unnocupied orbitals still exist here even if
    # they're empty, so it makes sense to pad here instead. No elements in the dataset exceed an
    # azimuthal of 3, so we only need to pad to length 10. Also: i'm realising here that the orbital
    # info will be unecessary if we have to pad here anyway
    return pad_tensor_to(torch.tensor(tuple(x[-1] for x in e)), 10)


def pad_tensor_to(t, length):
    return torch.nn.functional.pad(t, (0, length - t.shape[0]))


def load_dataset(filename):
    filepath = PUBCHEM_DIR / f"{filename}.pickle"
    with (filepath.parent / f"{filepath.stem}_x0{filepath.suffix}").open("rb") as f:
        x0 = pickle.load(f)
    with (filepath.parent / f"{filepath.stem}_x1{filepath.suffix}").open("rb") as f:
        x1 = pickle.load(f)
    with (filepath.parent / f"{filepath.stem}_y{filepath.suffix}").open("rb") as f:
        y = pickle.load(f)
    return x0, x1, y
