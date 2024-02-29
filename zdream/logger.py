import logging
from os import path
import os
import tkinter as tk

from PIL import Image
from loguru import logger
from typing import Dict, Tuple
from zdream.utils.misc import rmdir

from zdream.utils.model import DisplayScreen

# TODO - evolve the Logger into an IOHandler to save/load any type of data.
class Logger:

	'''
	Class responsible for logging in the three channels info, warn and error.

	NOTE: The logging methods can be easily overridden to log with other strategies 
	and technologies.
	'''

	def __init__(self, log_conf: Dict[str, str]) -> None:
		'''
		Initialize the logger with a specific target directory

		:param log_conf: mapping for experiment directory, title, name and version.
		:type out_dir: Dict[str, str]
		'''

		# Set empty target directory
		self._target_dir: str = self._get_target_dir(log_conf=log_conf)

		# Tinker main screen is mandatory, but we can hide it.
		self._main_screen = tk.Tk()
		self._main_screen.withdraw()
		
		# Initialize screen
		self._screens : Dict[str, DisplayScreen] = dict()

	# LOGGING

	def info(self,  mess: str): logging.info(mess)

	def warn(self,  mess: str): logging.warn(mess)

	def error(self, mess: str): logging.error(mess)

	# TARGET DIRECTORY

	def _get_target_dir(self, log_conf: Dict[str, str]) -> str:
		'''
		Return the target directory where to save results.

		:param log_conf: mapping for experiment directory, title, name and version.
		:type out_dir: Dict[str, str]
		'''

		# Parameters
		out_dir = log_conf['out_dir']
		title   = log_conf['title']
		name    = log_conf['name']
		version = log_conf['version']
	
		# Experiment directory
		exp_dir = path.join(out_dir, title, name)

		# If new version wasn't specified
		if not version:
			existing_dirs = os.listdir(exp_dir) if path.exists(exp_dir) else []
			if existing_dirs:
				# The new version is one plus the old version
				versions = sorted([int(version_name.split('-')[1]) for version_name in existing_dirs])
				version = versions[-1] + 1
			else:
				# First version
				version = 0
		
		version_name = f'{name}-{version}'
		target_dir   = path.join(exp_dir, version_name)

		# Check if it's overwriting and raise a warning in the case
		if path.exists(target_dir):
			self.warn(mess=f'The version will overwrite {target_dir}. Continue [y/n]')
			inp = input()
			# If continue we remove old directory
			if inp == 'y' or inp == 'Y':
				rmdir(target_dir)
			# otherwise we exit	
			else:
				self.error(" Choose an another version for non overwrite preexisting files. Exiting...")
				exit(1)

		return target_dir
	
	def create_target_dir(self):
	
		self.info(f"Creating  experiment directory {self.target_dir}")
		os.makedirs(self.target_dir, exist_ok=True)
		
	@property
	def target_dir(self) -> str:
		return self._target_dir


	# SCREEN

	def add_screen(self, screen_name: str, display_size: Tuple[int, int] = (400, 400)):
		'''
		Add a new screen with name and size. It raises a key error if that screen name already exists.
		'''

		if screen_name in self._screens.keys():
			err_msg = f'There already exists a screen with name {screen_name}'
			raise KeyError(err_msg)
		self._screens[screen_name] = DisplayScreen(title=screen_name, display_size=display_size)

	def update_screen(self, screen_name: str, image: Image.Image):
		'''
		Update the given screen with a new image.
		'''
		self._screens[screen_name].update(image=image)

	def remove_screen(self, screen_name: str):
		'''
		Remove a screen by ending its rendering and removing it from the list
		of logger screens. 
		'''

		if screen_name not in self._screens.keys():
			err_msg = f'Asked to remove a screen with name {screen_name}, but not in screen names.'
			raise KeyError(err_msg)

		# Stop rendering
		self._screens[screen_name].close()

		# Remove from the dictionary
		self._screens.pop(screen_name, None)

	def remove_all_screens(self):
		'''
		Remove all logger screens
		'''
		screen_names = list(self._screens.keys())
		for screen_name in screen_names:
			self.remove_screen(screen_name=screen_name)


class LoguruLogger(Logger):
	""" Logger overriding logger methods with `loguru` ones"""

	def __init__(self, log_conf: Dict[str, str]) -> None:
		super().__init__(log_conf)

		self._loguru_settings()

		
	def _loguru_settings(self):
		
		# Configure logger to write to both file and terminal
		log_file = path.join(self.target_dir, 'info.log')
		logger.add(log_file, level=0, enqueue=True) # Log to file
	
	def info(self, mess: str): logger.info(mess)
	def warn(self, mess: str): logger.warning(mess)
	def err (self, mess: str): logger.error(mess)