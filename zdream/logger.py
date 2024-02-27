import logging
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

	def __init__(self) -> None:

		# Tinker main screen is mandatory, but we can hide it.
		self._main_screen = tk.Tk()
		self._main_screen.withdraw()
		
		self._screens : Dict[str, DisplayScreen] = dict()

	def info(self,  mess: str): logging.info(mess)

	def warn(self,  mess: str): logging.warn(mess)

	def error(self, mess: str): logging.error(mess)

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

    def info(self, mess: str): logger.info(mess)
    def warn(self, mess: str): logger.warning(mess)
    def err (self, mess: str): logger.error(mess)