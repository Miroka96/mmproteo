import os

DEFAULT_PROJECT = "PXD010000"
DEFAULT_PROJECT_FILE_COUNT = 1175
DEFAULT_TEST_API = "2"

TESTS_PATH = os.path.join("tests")
RESOURCES_PATH = os.path.join(TESTS_PATH, "resources")

DATASET_PATH = os.path.join(TESTS_PATH, "..", "..", "datasets")
DEFAULT_PROJECT_DATASET_PATH = os.path.join(DATASET_PATH, DEFAULT_PROJECT)
os.makedirs(DEFAULT_PROJECT_DATASET_PATH, exist_ok=True)

MZML_FILE = "Biodiversity_A_cryptum_FeTSB_anaerobic_1_01Jun16_Pippin_16-03-39.mzML"
MZML_GZ_FILE = MZML_FILE + ".gz"
MZML_FILE_PATH = os.path.join(DEFAULT_PROJECT_DATASET_PATH, MZML_FILE)

MZID_FILE = "Biodiversity_A_cryptum_FeTSB_anaerobic_1_01Jun16_Pippin_16-03-39_msgfplus.mzid"
MZID_GZ_FILE = MZID_FILE + ".gz"
MZID_FILE_PATH = os.path.join(DEFAULT_PROJECT_DATASET_PATH, MZID_FILE)

MZMLID_FILE = "Biodiversity_A_cryptum_FeTSB_anaerobic_1_01Jun16_Pippin_16-03-39_mzmlid.parquet"
MZMLID_FILE_PATH = os.path.join(DEFAULT_PROJECT_DATASET_PATH, MZMLID_FILE)

RAW_FILE = "Biodiversity_A_cryptum_FeTSB_anaerobic_1_01Jun16_Pippin_16-03-39.raw"
MGF_FILE = "Biodiversity_A_cryptum_FeTSB_anaerobic_1_01Jun16_Pippin_16-03-39.mgf"
MGF_PARQUET_FILE = "Biodiversity_A_cryptum_FeTSB_anaerobic_1_01Jun16_Pippin_16-03-39.parquet"
