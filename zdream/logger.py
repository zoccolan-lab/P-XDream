import logging
from os import path
import os
import tkinter as tk

from PIL import Image
from loguru import logger
from typing import Dict, Tuple

from zdream.utils.model import DisplayScreen

# TODO - evolve the Logger into an IOHandler to save/load any type of data.
class Logger:

	'''
	Class responsible for logging in the three channels info, warn and error.

	NOTE: The logging methods can be easily overridden to log with other strategies 
	and technologies.
	'''

	def __init__(self, out_dir: str, exp_name: str, exp_version: str) -> None:
		'''
		Initialize the logger with a specific target directory

		:param out_dir: _description_
		:type out_dir: str
		:param exp_name: _description_
		:type exp_name: str
		:param exp_version: _description_
		:type exp_version: str
		'''

		# Set empty target directory
		self._target_dir: str = self._get_target_dir(out_dir=out_dir, exp_name=exp_name, exp_version=exp_version)

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

	def _get_target_dir(self, out_dir: str, exp_name: str, exp_version: str) -> str:
	
		# Experiment directory
		exp_dir = path.join(out_dir, exp_name)

		# Retrieve and create Experiment version directory
		same_version   = len(
			[filename for filename in os.listdir(exp_dir) if filename.startswith(exp_version)]
		) if os.path.exists(exp_dir) else 0

		version_number = '' if same_version == 0 else f'-{same_version}'
		version_name   = f'{exp_version}{version_number}'

		# Set target directory
		target_dir = path.join(exp_dir, version_name)

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

	def __init__(self, out_dir: str, exp_name: str, exp_version: str):
		super().__init__(out_dir, exp_name, exp_version)
	
	def info(self, mess: str): logger.info(mess)
	def warn(self, mess: str): logger.warning(mess)
	def err (self, mess: str): logger.error(mess)