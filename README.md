# MMProteo

`Mirko meets Proteomics`


## Getting started

### Requirements

* Python 3.7 or Python 3.8 (other versions haven't been tested yet)
* preferably Linux (other OSes haven't been tested yet)

If you want, set up a virtual environment to encapsulate this software and its dependencies:

```
virtualenv --python=/usr/bin/python3 venv
source venv/bin/activate
```

### Installation
#### ...from Pip

`pip install mmproteo`

#### ...from Source Code

```
cd mmproteo
python setup.py install
```

#### ...as editable Development Installation from Source Code

```
cd mmproteo
pip install -e .
```
This installation must be repeated whenever new files are created.

### Running MMProteo

The installation will make `mmproteo` available on your command line (and in your PATH),
so use this anywhere you want.

Alternatively, you can run the `mmproteo-runner.py` script in the mmproteo directory.

### Usage

Simply entering `mmproteo` will already show you the available parameters.
Using `mmproteo --help` you can get a better image of the functionality.
Currently, the following parameters are available:

```
$ mmproteo -h
usage: mmproteo [--count-failed-files] [--dummy-logger]
                [--file-extensions EXT] [--filter COLUMN[=!]=REGEX] [--help]
                [--log-file FILE] [--log-to-stdout] [--max-num-files N]
                [--no-count-skipped-files] [--no-fail-early]
                [--no-skip-existing] [--pride-project PROJECT]
                [--pride-version {1,2}] [--read-ions]
                [--shown-columns COLUMNS] [--storage-dir DIR]
                [--thermo-keep-running]
                [--thermo-output-format {imzml,mgf,mzml,parquet}]
                [--thread-count THREADS] [--verbose] [--version]
                COMMAND [COMMAND ...]

positional arguments:
  COMMAND               the list of actions to be performed on the repository.
                        Every action can only occur once. Duplicates are
                        dropped after the first occurrence.
                        convertraw  : convert all downloaded or extracted raw
                                      files or, if none were downloaded or
                                      extracted, those raw files in the data
                                      directory, into the given thermo output
                                      format using the ThermoRawFileParser.
                                      This command requires an accessible
                                      Docker installation.
                        download    : download files from a given project.
                        extract     : extract all downloaded archive files or,
                                      if none were downloaded, those in the
                                      data directory. Currently, the following
                                      archive formats are supported: "gz",
                                      "zip"
                        info        : request project information for a given
                                      project.
                        list        : list files and their attributes in a
                                      given project.
                        mgf2parquet : convert all downloaded, extracted, or
                                      converted mgf files into parquet format,
                                      or, if no files were previously
                                      processed, convert the mgf files in the
                                      data directory.
                        mz2parquet  : merge and convert all downloaded or
                                      extracted mzid and mzml files into
                                      parquet format or, if no files were
                                      previously processed, merge and convert
                                      the files in the data directory.
                        showconfig  : print all variables of the current run
                                      configuration.

optional arguments:
  --count-failed-files  Count failed files and do not just ignore them. This
                        is relevant for the max-num-files parameter. (default:
                        False)
  --dummy-logger        Use a simpler log format and log to stdout. (default:
                        False)
  --file-extensions EXT, -e EXT
                        a list of comma-separated allowed file extensions to
                        filter files for. Archive extensions will be
                        automatically appended. An empty list deactivates
                        filtering. Capitalization does not matter. (default: )
  --filter COLUMN[=!]=REGEX, -f COLUMN[=!]=REGEX
                        a filter condition for file filtering. The condition
                        must be of the form 'columnName[=!]=valueRegex'.
                        Therefore, the comparison operator can either be '=='
                        or '!='. The column name must not contain these
                        character patterns. The value will be interpreted
                        using Python's rules for regular expressions (from the
                        Python 're' package). This parameter can be given
                        multiple times to enforce multiple filters
                        simultaneously, meaning the filters will be logically
                        connected using a boolean 'and'. Boolean 'or'
                        operations can be specified as ' or ' within any
                        filter parameter, for example like this (representing
                        (a==1 or b==2) and (c==3 or (not d==4))): 'mmproteo -f
                        "a==1 or b==2" -f "c==3 or d!=4" list'. A condition
                        can be negated using '!=' instead of '=='. For the
                        filtering process, the filter columns will be
                        converted to strings. Non-existent columns will be
                        ignored. Capitalization matters. All these rules add
                        up to a conjunctive normal form. As some commands can
                        be pipelined to use previous results, there are also
                        the following special column names available:
                        ["converted_mgf_parquet_files",
                        "converted_mzmlid_parquet_files",
                        "converted_raw_files", "downloadLink",
                        "downloaded_files", "extracted_files", "fileName"]. An
                        empty list disables this filter. (default: [])
  --help, -h            Show this help message and exit.
  --log-file FILE, -l FILE
                        the name of the log file, relative to the download
                        directory. Set it to an empty string ("") to disable
                        file logging. (default: mmproteo.log)
  --log-to-stdout       Log to stdout instead of stderr. (default: False)
  --max-num-files N, -n N
                        the maximum number of files to be downloaded. Set it
                        to '0' to download all files. (default: 0)
  --no-count-skipped-files
                        Do not count skipped files and just ignore them. This
                        is relevant for the max-num-files parameter. (default:
                        True)
  --no-fail-early       Do not fail commands already on failed assertions. The
                        code will run until a real exception is encountered or
                        it even succeeds. (default: True)
  --no-skip-existing    Do not skip existing files. (default: True)
  --pride-project PROJECT, -p PROJECT
                        the name of the PRIDE project, e.g. 'PXD010000' from '
                        https://www.ebi.ac.uk/pride/ws/archive/peptide/list/pr
                        oject/PXD010000'. For some commands, this parameter is
                        required. (default: None)
  --pride-version {1,2}, -i {1,2}
                        an API version for the PRIDE interactions. Only the
                        specified versions will be used. This parameter can be
                        given multiple times to allow multiple different API
                        versions, one version per parameter appearance. The
                        order of occurring API versions will be considered
                        until the first API request fulfills its job. Every
                        version should appear at most once. Duplicates are
                        dropped after the first occurrence. An empty list
                        (default) uses all api versions in the following
                        order: ["2", "1"] (default: [])
  --read-ions           Read ion names when parsing spectra. This only applies
                        to MGF file parsing, but might resolve file parsing
                        exceptions. (default: False)
  --shown-columns COLUMNS, -c COLUMNS
                        a list of comma-separated column names. Some commands
                        show their results as tables, so their output columns
                        will be limited to those in this list. An empty list
                        deactivates filtering. Capitalization matters.
                        (default: )
  --storage-dir DIR, -d DIR
                        the name of the directory, in which the downloaded
                        files and the log file will be stored. (default: .)
  --thermo-keep-running
                        Keep the ThermoRawFileParser Docker container running
                        after conversion. This can speed up batch processing
                        and ease debugging. (default: False)
  --thermo-output-format {imzml,mgf,mzml,parquet}
                        the output format into which the raw file will be
                        converted. This parameter only applies to the
                        convertraw command. (default: mgf)
  --thread-count THREADS
                        the number of threads to use for parallel processing.
                        Set it to '0' to use as many threads as there are CPU
                        cores. Setting the number of threads to '1' disables
                        parallel processing. (default: 1)
  --verbose, -v         Increase output verbosity to debug level. (default:
                        False)
  --version             Show the version of this software.
```

There are multiple commands (positional arguments) as well as several optional arguments available.
A single command doesn't require or use all the available arguments. However, one can provide multiple
commands (separated by spaces), which will all use their share of the given arguments.

### Examples

#### Request Project Information for the Project PXD010000

`mmproteo --pride-project PXD010000 info`

#### List all Information about 2 Files of Project PXD010000

`mmproteo --pride-project PXD010000 --max-num-files 2 list`

#### Show selected Information about 5 RAW or mzML files of Project PXD010000
```
$ mmproteo --pride-project PXD010000 --max-num-files 5 --shown-columns "projectAccessions,fileName,fileSizeBytes" --file-extensions "raw,mzml" list
2021-03-04 22:13:41,374 - MMProteo: Logging to file './mmproteo.log' and to <stderr>
2021-03-04 22:13:41,374 - MMProteo: Requesting list of project files using API version 2 from https://www.ebi.ac.uk/pride/ws/archive/v2/files/byProject?accession=PXD010000
2021-03-04 22:13:42,280 - MMProteo: Received project file list using API version 2 with 1175 files for project "PXD010000"
2021-03-04 22:13:42,280 - MMProteo: Filtering files based on the following required file extensions ["mzml", "raw"] and the following optional file extensions ["gz", "zip"]
2021-03-04 22:13:42,288 - MMProteo: Showing only the top 5 entries because of the max_num_files parameter
2021-03-04 22:13:42,289 - MMProteo: Limiting the shown columns according to the shown_columns parameter
  projectAccessions                                                                  fileName  fileSizeBytes
0       [PXD010000]  Biodiversity_A_cryptum_FeTSB_anaerobic_1_01Jun16_Pippin_16-03-39.mzML.gz      281892753
1       [PXD010000]      Biodiversity_A_cryptum_FeTSB_anaerobic_1_01Jun16_Pippin_16-03-39.raw      386157431
2       [PXD010000]  Biodiversity_A_cryptum_FeTSB_anaerobic_2_01Jun16_Pippin_16-03-39.mzML.gz      298698862
3       [PXD010000]      Biodiversity_A_cryptum_FeTSB_anaerobic_2_01Jun16_Pippin_16-03-39.raw      410279729
4       [PXD010000]  Biodiversity_A_cryptum_FeTSB_anaerobic_3_01Jun16_Pippin_16-03-39.mzML.gz      276592897
```
Recognize, that the archive extension `.gz` of the mzML files as well as the capitalization of `mzML` 
is ignored while filtering the files.

#### List 5 files with a file size below 100 megabytes or a 'gz' file extension

`mmproteo --pride-project PXD010000 --max-num-files 5 --filter "fileSizeBytes==.{1,8} or fileName==.*\.gz" list`

The filter parameter uses regular expressions to match column values. 
You can check the syntax on the ['re' homepage](https://docs.python.org/3/library/re.html).

## Tests

Right now, there is only a limited number of tests. Most functionality is only tested using integration tests, because
the methods became too quickly too many.
Still, to run the tests you will need mmproteo installed, preferably as edible development installation 
(run `pip install -e .` in mmproteo base directory). Furthermore, pytest is required. To directly install all packages required for development 
and extended usage of this project, run `pip install -r requirements.txt` in the root of this project.

Finally, to run the tests, `cd` into the test directory (`tests`) and run `pytest` there. Unfortunately, 
depending on your Internet connection and CPU speed, some tests might take several minutes, because I don't know of a 
really nice way of testing a downloader/converter without downloading/converting stuff. Furthermore, some tests 
currently require `docker` being available and usable to run raw-file conversions.

Good Luck!
