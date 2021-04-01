#!/usr/bin/python3

import os
import subprocess
import time

DEFAULT_PROJECT = "PXD010000"
FAIR_USE_DELAY_SECONDS = 0.5
RESOURCE_DIR = os.path.sep.join(["tests", "resources", ""])


def _store_command_output(command: str, filename: str, include_stderr: bool = True) -> None:
    system_command = f"{command} > {filename}"
    if include_stderr:
        system_command += " 2>&1"
    subprocess.run(system_command, shell=True)


def _compare_stdout_with_file(command: str, filename: str, include_stderr: bool = True) -> None:
    if include_stderr:
        command += " 2>&1"

    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    stdout = process.stdout.decode('utf-8')
    stdout_lines = stdout.split("\n")

    stdout_lines = stdout_lines[:-1]  # remove empty trailing line

    with open(filename) as file:
        file_lines = file.readlines()
    file_lines = [line.rstrip("\n") for line in file_lines]

    assert len(stdout_lines) == len(file_lines), "outputs do not have the same lengths"
    for received, expected in zip(stdout_lines, file_lines):
        assert received == expected, "found lines that do not match"


def store_mmproteo_output():
    _store_command_output(command="mmproteo",
                          filename=RESOURCE_DIR + "mmproteo_output.txt")


def test_mmproteo_output():
    _compare_stdout_with_file(command="mmproteo",
                              filename=RESOURCE_DIR + "mmproteo_output.txt")


def store_mmproteo_h_output():
    _store_command_output(command="mmproteo -h",
                          filename=RESOURCE_DIR + "mmproteo_h_output.txt")


def test_mmproteo_h_output():
    _compare_stdout_with_file(command="mmproteo -h",
                              filename=RESOURCE_DIR + "mmproteo_h_output.txt")


def store_mmproteo_p_info_output():
    _store_command_output(command=f"mmproteo -p {DEFAULT_PROJECT} info",
                          filename=RESOURCE_DIR + "mmproteo_p_info_output.txt",
                          include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def test_mmproteo_p_info_output():
    _compare_stdout_with_file(command=f"mmproteo -p {DEFAULT_PROJECT} info",
                              filename=RESOURCE_DIR + "mmproteo_p_info_output.txt",
                              include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def store_mmproteo_p_list_output():
    _store_command_output(command=f"mmproteo -p {DEFAULT_PROJECT} list",
                          filename=RESOURCE_DIR + "mmproteo_p_list_output.txt",
                          include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def test_mmproteo_p_list_output():
    _compare_stdout_with_file(command=f"mmproteo -p {DEFAULT_PROJECT} list",
                              filename=RESOURCE_DIR + "mmproteo_p_list_output.txt",
                              include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def store_mmproteo_p_n_list_output():
    _store_command_output(command=f"mmproteo -p {DEFAULT_PROJECT} -n 10 list",
                          filename=RESOURCE_DIR + "mmproteo_p_n_list_output.txt",
                          include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def test_mmproteo_p_n_list_output():
    _compare_stdout_with_file(command=f"mmproteo -p {DEFAULT_PROJECT} -n 10 list",
                              filename=RESOURCE_DIR + "mmproteo_p_n_list_output.txt",
                              include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def store_mmproteo_p_e_mzid_list_output():
    _store_command_output(command=f"mmproteo -p {DEFAULT_PROJECT} -e mzid list",
                          filename=RESOURCE_DIR + "mmproteo_p_e_mzid_list_output.txt",
                          include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def test_mmproteo_p_e_mzid_list_output():
    _compare_stdout_with_file(command=f"mmproteo -p {DEFAULT_PROJECT} -e mzid list",
                              filename=RESOURCE_DIR + "mmproteo_p_e_mzid_list_output.txt",
                              include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def store_mmproteo_p_e_gz_list_output():
    _store_command_output(command=f"mmproteo -p {DEFAULT_PROJECT} -e gz list",
                          filename=RESOURCE_DIR + "mmproteo_p_e_gz_list_output.txt",
                          include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def test_mmproteo_p_e_gz_list_output():
    _compare_stdout_with_file(command=f"mmproteo -p {DEFAULT_PROJECT} -e gz list",
                              filename=RESOURCE_DIR + "mmproteo_p_e_gz_list_output.txt",
                              include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def store_mmproteo_p_e_gz_mzid_list_output():
    _store_command_output(command=f"mmproteo -p {DEFAULT_PROJECT} -e gz,mzid list",
                          filename=RESOURCE_DIR + "mmproteo_p_e_gz_mzid_list_output.txt",
                          include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def test_mmproteo_p_e_gz_mzid_list_output():
    _compare_stdout_with_file(command=f"mmproteo -p {DEFAULT_PROJECT} -e gz,mzid list",
                              filename=RESOURCE_DIR + "mmproteo_p_e_gz_mzid_list_output.txt",
                              include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def store_mmproteo_p_c_list_output():
    _store_command_output(command=f"mmproteo -p {DEFAULT_PROJECT} -c fileName,downloadLink list",
                          filename=RESOURCE_DIR + "mmproteo_p_c_list_output.txt",
                          include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def test_mmproteo_p_c_list_output():
    _compare_stdout_with_file(command=f"mmproteo -p {DEFAULT_PROJECT} -c fileName,downloadLink list",
                              filename=RESOURCE_DIR + "mmproteo_p_c_list_output.txt",
                              include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def store_mmproteo_p_n_e_c_list_output():
    _store_command_output(command=f"mmproteo -p {DEFAULT_PROJECT} -n 100 -e raw,mzml,gz -c fileName,downloadLink list",
                          filename=RESOURCE_DIR + "mmproteo_p_n_e_c_list_output.txt",
                          include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def test_mmproteo_p_n_e_c_list_output():
    _compare_stdout_with_file(command=f"mmproteo -p {DEFAULT_PROJECT} -n 100 -e raw,mzml,gz "
                                      f"-c fileName,downloadLink list",
                              filename=RESOURCE_DIR + "mmproteo_p_n_e_c_list_output.txt",
                              include_stderr=False)
    time.sleep(FAIR_USE_DELAY_SECONDS)


def store_mmproteo_showconfig_output():
    _store_command_output(command="mmproteo showconfig",
                          filename=RESOURCE_DIR + "mmproteo_showconfig_output.txt",
                          include_stderr=False)


def test_mmproteo_showconfig_output():
    _compare_stdout_with_file(command="mmproteo showconfig",
                              filename=RESOURCE_DIR + "mmproteo_showconfig_output.txt",
                              include_stderr=False)


if __name__ == '__main__':
    # run this file as script to recreate the expected outputs
    store_mmproteo_output()
    store_mmproteo_h_output()
    store_mmproteo_p_info_output()
    store_mmproteo_p_list_output()
    store_mmproteo_p_n_list_output()
    store_mmproteo_p_e_mzid_list_output()
    store_mmproteo_p_e_gz_list_output()
    store_mmproteo_p_e_gz_mzid_list_output()
    store_mmproteo_p_c_list_output()
    store_mmproteo_p_n_e_c_list_output()
    store_mmproteo_showconfig_output()
