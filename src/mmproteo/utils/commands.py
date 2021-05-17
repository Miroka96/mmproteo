from typing import Dict, List

from mmproteo.utils import filters, log, pride, utils, visualization
from mmproteo.utils.config import Config
from mmproteo.utils.formats import archives


class AbstractCommand:
    def get_command(self) -> str:
        raise NotImplementedError

    def get_description(self) -> str:
        raise NotImplementedError

    def run(self, config: Config, logger: log.Logger = log.DEFAULT_LOGGER) -> None:
        raise NotImplementedError

    def validate(self, config: Config, logger: log.Logger = log.DEFAULT_LOGGER) -> None:
        pass


class DownloadCommand(AbstractCommand):

    def get_command(self) -> str:
        return "download"

    def get_description(self) -> str:
        return "download files from a given project."

    def run(self, config: Config, logger: log.Logger = log.DEFAULT_LOGGER) -> None:
        project_files = config.get_project_files()
        downloaded_files = pride.download(project_name=config.pride_project if project_files is None else None,
                                          project_files=project_files,
                                          valid_file_extensions=config.valid_file_extensions,
                                          max_num_files=config.max_num_files,
                                          column_filter=config.column_filter,
                                          download_dir=config.storage_dir,
                                          skip_existing=config.skip_existing,
                                          count_failed_files=config.count_failed_files,
                                          count_skipped_files=config.count_skipped_files,
                                          file_name_column=config.default_file_name_column,
                                          download_link_column=config.default_download_link_column,
                                          downloaded_files_column=config.default_downloaded_files_column,
                                          api_versions=config.pride_versions,
                                          thread_count=config.thread_count,
                                          logger=logger)
        if downloaded_files is None:
            return

        downloaded_files = downloaded_files.dropna(subset=[config.default_downloaded_files_column])

        config.cache_processed_files(data_df=downloaded_files)
        visualization.print_df(df=downloaded_files,
                               shown_columns=config.shown_columns + [config.default_downloaded_files_column],
                               urlencode_columns=[config.default_download_link_column],
                               logger=logger)

    def validate(self, config: Config, logger: log.Logger = log.DEFAULT_LOGGER) -> None:
        config.require_pride_project(logger=logger)

        if len(config.valid_file_extensions) != 0 and config.max_num_files % len(config.valid_file_extensions) != 0:
            logger.info(
                "max-num-files should be a multiple of the number of valid_file_extensions to make sure that "
                "files that belong together are also downloaded together")


class InfoCommand(AbstractCommand):

    def get_command(self) -> str:
        return "info"

    def get_description(self) -> str:
        return "request project information for a given project."

    def run(self, config: Config, logger: log.Logger = log.DEFAULT_LOGGER) \
            -> None:
        assert config.pride_project is not None, \
            "pride project existence should have been checked using the " \
            "validate function"
        project_info = pride.get_project_info(project_name=config.pride_project,
                                              api_versions=config.pride_versions,
                                              logger=logger)
        if project_info is None:
            return
        print(project_info)

    def validate(self, config: Config, logger: log.Logger = log.DEFAULT_LOGGER) \
            -> None:
        config.require_pride_project(logger=logger)


class ListCommand(AbstractCommand):

    def get_command(self) -> str:
        return "list"

    def get_description(self) -> str:
        return "list files and their attributes in a given project."

    def run(self, config: Config, logger: log.Logger = log.DEFAULT_LOGGER) -> None:
        project_files = config.get_project_files()

        if project_files is None:
            return

        filtered_files = filters.filter_files_df(files_df=project_files,
                                                 file_extensions=config.valid_file_extensions,
                                                 column_filter=config.column_filter,
                                                 sort=True,
                                                 logger=logger)

        visualization.print_df(df=filtered_files,
                               max_num_files=config.max_num_files,
                               shown_columns=config.shown_columns,
                               urlencode_columns=[config.default_download_link_column],
                               logger=logger)

    def validate(self, config: Config, logger: log.Logger = log.DEFAULT_LOGGER) -> None:
        config.require_pride_project(logger=logger)


class ExtractCommand(AbstractCommand):

    def get_command(self) -> str:
        return "extract"

    def get_description(self) -> str:
        return "extract all downloaded archive files or, if none were downloaded, those in the data directory. " \
               "Currently, the following archive formats are supported: " + \
               archives.get_string_of_extractable_file_extensions()

    def run(self, config: Config, logger: log.Logger = log.DEFAULT_LOGGER) -> None:
        files = config.get_processed_files(config.default_downloaded_files_column)

        if len(files) == 0:
            files = utils.list_files_in_directory(config.storage_dir)

        extracted_files = archives.extract_files(filenames=files,
                                                 skip_existing=config.skip_existing,
                                                 max_num_files=config.max_num_files,
                                                 count_failed_files=config.count_failed_files,
                                                 count_skipped_files=config.count_skipped_files,
                                                 thread_count=config.thread_count,
                                                 column_filter=config.column_filter,
                                                 keep_null_values=False,
                                                 pre_filter_files=True,
                                                 logger=logger)

        result_df = config.cache_processed_files(data_list=extracted_files,
                                                 column_names=config.default_extracted_files_column)
        visualization.print_df(df=result_df, logger=logger)


class ConvertRawCommand(AbstractCommand):

    def get_command(self) -> str:
        return "convertraw"

    def get_description(self) -> str:
        return "convert all downloaded or extracted raw files or, if none were downloaded or extracted, " \
               "those raw files in the data " \
               "directory, into the given thermo output format using the ThermoRawFileParser. " \
               "This command requires an accessible Docker installation."

    def run(self, config: Config, logger: log.Logger = log.DEFAULT_LOGGER) -> None:
        files = config.get_processed_files(config.default_downloaded_files_column,
                                           config.default_extracted_files_column)

        if len(files) == 0:
            files = utils.list_files_in_directory(config.storage_dir)

        from mmproteo.utils.formats import raw
        raw.start_thermo_docker_container(storage_dir=config.storage_dir,
                                          thermo_docker_container_name=Config.default_thermo_docker_container_name,
                                          thermo_docker_image=Config.default_thermo_docker_image,
                                          thermo_start_container_command_template=Config.
                                          default_thermo_start_container_command_template,
                                          logger=logger)

        converted_files = raw.convert_raw_files(filenames=files,
                                                output_format=config.thermo_output_format,
                                                skip_existing=config.skip_existing,
                                                max_num_files=config.max_num_files,
                                                count_failed_files=config.count_failed_files,
                                                count_skipped_files=config.count_skipped_files,
                                                thread_count=config.thread_count,
                                                column_filter=config.column_filter,
                                                keep_null_values=False,
                                                pre_filter_files=True,
                                                thermo_docker_container_name=Config.
                                                default_thermo_docker_container_name,
                                                thermo_exec_command=Config.default_thermo_exec_command,
                                                logger=logger)

        result_df = config.cache_processed_files(converted_files, config.default_converted_raw_files_column)

        if not config.thermo_keep_container_running:
            raw.stop_thermo_docker_container(
                thermo_docker_container_name=Config.default_thermo_docker_container_name,
                logger=logger
            )

        visualization.print_df(df=result_df, logger=logger)

    def validate(self, config: Config, logger: log.Logger = log.DEFAULT_LOGGER) -> None:
        from mmproteo.utils.formats import raw
        raw.assert_valid_thermo_output_format(output_format=config.thermo_output_format,
                                              logger=logger)


class Mgf2ParquetCommand(AbstractCommand):

    def get_command(self) -> str:
        return "mgf2parquet"

    def get_description(self) -> str:
        return "convert all downloaded, extracted, or converted mgf files into parquet format, or, " \
               "if no files were previously processed, convert the mgf files in the data directory."

    def run(self, config: Config, logger: log.Logger = log.DEFAULT_LOGGER) -> None:
        files = config.get_processed_files(config.default_downloaded_files_column,
                                           config.default_extracted_files_column,
                                           config.default_converted_raw_files_column)

        if len(files) == 0:
            files = utils.list_files_in_directory(config.storage_dir)

        from mmproteo.utils.formats import mgf
        mgf_parquet_files = mgf.convert_mgf_files_to_parquet(filenames=files,
                                                             skip_existing=config.skip_existing,
                                                             max_num_files=config.max_num_files,
                                                             count_failed_files=config.count_failed_files,
                                                             count_skipped_files=config.count_skipped_files,
                                                             thread_count=config.thread_count,
                                                             column_filter=config.column_filter,
                                                             keep_null_values=False,
                                                             pre_filter_files=True,
                                                             logger=logger)

        result_df = config.cache_processed_files(mgf_parquet_files, config.default_mgf_parquet_files_column)
        visualization.print_df(df=result_df, logger=logger)


class Mz2ParquetCommand(AbstractCommand):
    def get_command(self) -> str:
        return "mz2parquet"

    def get_description(self) -> str:
        return "merge and convert all downloaded or extracted mzid and mzml files into parquet format" \
               " or, if no files were previously processed, merge and convert the files in the data directory."

    def run(self, config: Config, logger: log.Logger = log.DEFAULT_LOGGER) -> None:
        files = config.get_processed_files(
            config.default_downloaded_files_column, config.default_extracted_files_column)

        if len(files) == 0:
            files = utils.list_files_in_directory(
                config.storage_dir)

        from mmproteo.utils.formats import mz
        mzmlid_parquet_files = mz.merge_mzml_and_mzid_files_to_parquet(filenames=files,
                                                                       skip_existing=config.skip_existing,
                                                                       max_num_files=config.max_num_files,
                                                                       count_failed_files=config.count_failed_files,
                                                                       count_skipped_files=config.count_skipped_files,
                                                                       thread_count=config.thread_count,
                                                                       column_filter=config.column_filter,
                                                                       logger=logger)

        result_df = config.cache_processed_files(mzmlid_parquet_files, config.default_mzmlid_parquet_files_column)
        visualization.print_df(df=result_df, logger=logger)


class ShowConfigCommand(AbstractCommand):
    def get_command(self) -> str:
        return "showconfig"

    def get_description(self) -> str:
        return "print all variables of the current run configuration."

    def run(self, config: Config, logger: log.Logger = log.DEFAULT_LOGGER) -> None:
        print(config)


class CommandDispatcher:
    def __init__(self) -> None:
        self.__commands: Dict[str, AbstractCommand] = dict()

    def register(self, command: AbstractCommand) -> None:
        if command.get_command() in self.__commands.keys():
            raise ValueError("Command is already registered")
        self.__commands[command.get_command()] = command

    def get_command_names(self) -> List[str]:
        return sorted(self.__commands.keys())

    @staticmethod
    def _pad_command(command: str, width: int) -> str:
        return command.ljust(width)

    def get_command_descriptions_str(self) -> str:
        longest_command_length = max([len(command) for command in self.get_command_names()])

        return '\n'.join(
            [self._pad_command(command.get_command(), longest_command_length) + " : " + command.get_description()
             for command in self.__commands.values()])

    def get_command(self, command_name: str) -> AbstractCommand:
        if command_name not in self.__commands.keys():
            raise NotImplementedError("%s is no known command")
        return self.__commands[command_name]

    def dispatch_commands(self,
                          config: Config,
                          catch_validation_warnings: bool = True,
                          catch_run_warnings: bool = True,
                          logger: log.Logger = log.DEFAULT_LOGGER) -> None:
        assert config.commands is not None, "commands are missing"
        commands = [self.get_command(command_name) for command_name in config.commands]

        for command in commands:
            if catch_validation_warnings:
                try:
                    command.validate(config=config, logger=logger)
                except log.LoggedWarningException:
                    pass
            else:
                command.validate(config=config, logger=logger)

        for command in commands:
            if catch_run_warnings:
                try:
                    command.run(config=config, logger=logger)
                except log.LoggedWarningException:
                    pass
            else:
                command.run(config=config, logger=logger)


DISPATCHER = CommandDispatcher()
DISPATCHER.register(ConvertRawCommand())
DISPATCHER.register(DownloadCommand())
DISPATCHER.register(ExtractCommand())
DISPATCHER.register(InfoCommand())
DISPATCHER.register(ListCommand())
DISPATCHER.register(Mgf2ParquetCommand())
DISPATCHER.register(Mz2ParquetCommand())
DISPATCHER.register(ShowConfigCommand())
