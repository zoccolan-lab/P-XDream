import os
import tkinter as tk
import logging
from os     import path
from typing import Dict, Tuple

from PIL    import Image
import loguru

from zdream.utils.io_  import rmdir
from zdream.utils.model import DisplayScreen

class Logger:
	'''
	Class responsible for 
	- logging in the channels info, warn and error;
	- handling display screens;
	- organize target directory for saving results.
	'''

	def __init__(self, conf: Dict[str, str] | None = None) -> None:
		'''
		Initialize the logger with a possible specific target directory.

		:param conf: Mapping for experiment directory, title, name and version.
		                 If not specified the target directory is not set and the object
						 only serves for logging and display purposes.
		:type conf: Dict[str, str] | None
		'''

		# Set target directory if specified
		self._target_dir: str = self._get_target_dir(conf=conf) if conf else ''

		# Create the mandatory Tinker, which we hide.
		self._main_screen = tk.Tk()
		self._main_screen.withdraw()
		
		# Initialize screen dictionary
		self._screens : Dict[str, DisplayScreen] = dict()

		# Public prefix
		self.prefix = ''

	# LOGGING
		
	# NOTE: The logging methods uses python default logging, 
	#	    but can easily  be overridden to log with other strategies and technologies.

	def  info(self,  mess: str): self._info(mess=f'{self.prefix}{mess}')
	def _info(self,  mess: str): logging.info(mess)

	def  warn(self,  mess: str): self._warn(mess=f'{self.prefix}{mess}')
	def _warn(self,  mess: str): logging.warn(mess)

	def  error(self, mess: str): self._error(mess=f'{self.prefix}{mess}')
	def _error(self, mess: str): logging.error(mess)

	# TARGET DIRECTORY

	def _get_target_dir(self, conf: Dict[str, str]) -> str:
		'''
		Return the target directory for saving experiment results.

		In the case version key is specified it logs at the level of experiment versioning.
		If not specified it logs at the level of the specific experiment.

		:param conf: Mapping for experiment directory, title, name and version.
		:type conf: Dict[str, str]
		:return: Target directory.
		:rtype: str
		'''

		# Extract parameters
		out_dir = conf['out_dir']
		title   = conf['title']
		name    = conf['name']

		# Experiment directory
		exp_dir = path.join(out_dir, title, name)

		# If the version is not a key the target directory
		# is the experiment directory
		if not 'version' in conf:
			return exp_dir

		# If the version not a key the target directory
		# is the version directory
		version = conf['version']

		# Compute new version if it wasn't specified
		if not version:

			# List previously existing versions
			existing_versions = [
				file for file in os.listdir(exp_dir) if not '.' in file # TODO better matching
			] if path.exists(exp_dir) else []

			# If at least one exists, use the last version plus one
			if existing_versions:
				versions = sorted([
					int(version_name.split('-')[1]) for version_name in existing_versions
				])
				version = versions[-1] + 1

			# Otherwise set the first version as the zero one
			else:
				version = 0
		
		# Build target directory path
		version_name = f'{name}-{version}'
		target_dir   = path.join(exp_dir, version_name)

		# Check if it the version is overwriting an existing one
		# In the case, ask confirm for overwriting
		# NOTE This is possible because we allow an arbitrary input version specification
		if path.exists(target_dir):

			self.warn(mess=f'The new new version will overwrite {target_dir}. Continue [y/n]')
			inp = input()

			# If continue, remove old directory
			if inp == 'y' or inp == 'Y':
				rmdir(target_dir)

			# If not continue, exit
			else:
				self.error("Choose an another version for non overwrite preexisting files. Exiting...")
				exit(1)

		return target_dir
	
	def create_target_dir(self):
		''' Creates the experiment directory '''
	
		self.info(f"Creating experiment directory {self.target_dir}")
		os.makedirs(self.target_dir, exist_ok=True)
		
	@property
	def target_dir(self) -> str:
		'''
		Returns experiment target directory.
		If it was not provided during initialization raises an error.
		'''

		if self._target_dir:
			return self._target_dir
		raise ValueError('Target directory was not provided during logger initialization. '\
				         'Use the parameter `config` during initialization to set one.')

	# SCREEN

	def add_screen(self, screen_name: str, display_size: Tuple[int, int] = (400, 400)):
		'''
		Add a new screen with name and size. It raises a key error if that screen name already exists.
		
		:param screen_name: Name identifier for the new screen name.
		:type screen_name: str.
		:param display_size: New screen display size in pixels, defaults to (400, 400).
		:type display_size: Tuple[int, int]
		'''

		if screen_name in self._screens:

			err_msg = f'There already exists a screen with name {screen_name}'
			raise KeyError(err_msg)
		
		self._screens[screen_name] = DisplayScreen(title=screen_name, display_size=display_size)

	def update_screen(self, screen_name: str, image: Image.Image):
		'''
		Update a display screen with a new image. It raises a key error if that screen name doesn't exist.
		
		:param screen_name: Name identifier for the new screen name.
		:type screen_name: str.
		:param image: Image to update screen frame.
		:type image: Image.Image
		'''

		try:
			self._screens[screen_name].update(image=image)

		except KeyError:
			raise KeyError(f'Screen {screen_name} not present in screens {self._screens.keys()}.')

	def remove_screen(self, screen_name: str):
		'''
		Remove a screen by first ending its rendering and then removing it from the list of screens.
		It raises a key error if that screen name doesn't exist.
		
		:param screen_name: Name identifier for the new screen name.
		:type screen_name: str.
		'''

		# Stop rendering
		try:
			self._screens[screen_name].close()

		except KeyError:
			raise KeyError( f'Trying to remove screen {screen_name}, but not present in screens {self._screens.keys()}.')

		# Remove from the dictionary
		self._screens.pop(screen_name, None)

	def remove_all_screens(self):
		'''
		Remove all display screens.
		'''

		# This is necessary to prevent dictionary on-loop changes
		screen_names = list(self._screens.keys())

		for screen_name in screen_names:
			self.remove_screen(screen_name=screen_name)


class LoguruLogger(Logger):
	""" Logger using `loguru` technology to log both on terminal and file """

	_serial_number = 0

	def __init__(self, conf: Dict[str, str] | None = None, on_file: bool = True) -> None:
		'''
		Initialize the logger with a possible specific target directory.
		In the case the target directory is specified and the `on_file` flag is active,
		it also logs to file.

		:param conf: Mapping for experiment directory, title, name and version.
		                 If not specified the target directory is not set and the object
						 only serves for logging and display purposes.
		:type conf: Dict[str, str] | None
		:param on_file: If to log on file, defaults to True
		:type on_file: bool, optional
		'''

		super().__init__(conf=conf)

		self._id = self._serial_number
		LoguruLogger._serial_number += 1

		self._logger = loguru.logger.bind(id=self._id)
		
		# File logging
		if on_file and conf:

			log_file = path.join(self.target_dir, 'info.log')
			self._logger.add(
				log_file, level=0, enqueue=True, 
				filter=lambda x: x['extra']['id']==self._id
			)

		# Warning if on_file but no target directory.
		elif on_file:

			self.warn('The `on_file` flag was activated but no target directory was specified')
	
	# Overriding logging methods with `loguru` specific ones
	def _info(self, mess: str): self._logger.info(mess)
	def _warn(self, mess: str): self._logger.warning(mess)
	def _err (self, mess: str): self._logger.error(mess)


class MutedLogger(Logger):
	''' Trivial logger with for non-logging '''

	def __init__(self) -> None:
		''' Use None configuration for non specifying a target directory '''

		super().__init__(conf=None)
	
	# Override for no logging
	def _info(self, mess: str): pass
	def _warn(self, mess: str): pass
	def _err (self, mess: str): pass