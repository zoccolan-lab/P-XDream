'''
This module contains the implementation of the Logger class, which is responsible for
	- logging in the three channels INFO, WARN and ERROR;
	- handling display screens;
	- organize target directory for saving results.

The Logger class is a base class that can be extended to implement different logging strategies.
The following are the implemented classes:
- Logger: Base class for logging that uses the Python `logging` module.
- LoguruLogger: Logger using `loguru` technology to log both on terminal and file.
- SilentLogger: Trivial logger with for non-logging.

The class also provides the implementation of the `DisplayScreen` class, which is a screen to display and update images.
'''

from __future__ import annotations

import os
from os import path
import logging
import tkinter as tk
from typing import Callable, Dict, List, Tuple
from rich.console import Console

from PIL import Image, ImageTk
import loguru

from pxdream.utils.parameters import ArgParams, ParamConfig

from .io_   import rmdir
import os
from os import path
import logging
from typing import Callable, Dict, List
from PIL import Image
import loguru
from .io_   import rmdir


class Logger():
	'''
	Class responsible for 
		- logging in the channels info, warn and error;
		- handling display screens;
		- organize foldering for saving results.
	'''

	# Rich console progress bar 
	CONSOLE = Console(color_system=None, stderr=False)

	def __init__(self, path: Dict[str, str] | str  = '.') -> None:
		'''
		Initialize the logger with a possible specific target directory.

		:param path: Path where to save the experiment results.
			It supports to files.
			- Dict[str, str]:   dictionary specifiying the target directory hierarchy based 
                                on experiment directory, title, name and version.
            - str:              target directory path.
		:type path: Dict[str, str] | None
		'''

		# Formatting function to dynamically manipulate string to logging information
		# e.g. added a fixed prefix
		self._formatting: Callable[[str], str] = lambda x: x

		# Set target directory
		# depending on the type of input
		self._dir: str = self._get_dir(conf=path) if isinstance(path, dict) else path
		
		# Initialize screen dictionary
		self._screens : Dict[str, DisplayScreen] = dict()

	@classmethod
	def from_conf(cls, conf: ParamConfig) -> 'Logger': 
		'''
		Initialize the logger from parameter configuration
		The parameter requires specifications for output directory and experiment
			- title: it typically refers to the experiment class it is run with, for example `MaximizeActivity`.
			- name: it refers to the specific purpose the experiment is run with.
			- version: it refers to the versioning of the experiment, for example by changing the hyperparameters.

		:param conf: Parameter configuration for the experiment.
		:type conf: ParamConfig
		:return: Mapping for experiment directory, title, name and version.
		:rtype: Dict[str, str]
		'''

		try: 
		
			return Logger(
				path= {
				'out_dir': str(conf[ArgParams.OutputDirectory  .value]),
				'title'  : str(conf[ArgParams.ExperimentTitle  .value]),
				'name'   : str(conf[ArgParams.ExperimentName   .value]),
				'version': str(conf[ArgParams.ExperimentVersion.value])
				}
			)
		
		except KeyError as e:
			raise KeyError(f'Cannot create Logger from configuration as some keys are missing {e}')

	# --- FORMATTING ---

	@property
	def formatting(self) -> Callable[[str], str]:
		'''
		Returns the formatting function for dynamically manipulating the string 
		before logging information.
		'''
		return self._formatting
	
	@formatting.setter
	def formatting(self, formatting: Callable[[str], str]):
		'''
		Set the formatting function for dynamically manipulating the string 
		before logging information.

		:param formatting: Formatting function.
		:type formatting: Callable[[str], str]
		'''
		self._formatting = formatting

	def reset_formatting(self): self.formatting = lambda x: x
	''' Reset the formatting function to the default one. '''

	# --- LOGGING ---
	
	# NOTE: The public logging methods append the prefix to the message which is passed to
	#       its private method version for the actual logging.
	#	    Subclasses that intend to log with other strategies and technologies should
	#       override the private ones.
	
	def  info(self,  msg: str): self._info(msg=self.formatting(msg))
	def _info(self,  msg: str): logging.info(msg=msg)
	
	def  warn(self,  msg: str): self._warn(msg=self.formatting(msg))
	def _warn(self,  msg: str): logging.warning(msg=msg)
	
	def  error(self, msg: str): self._error(msg=self.formatting(msg))
	def _error(self, msg: str): logging.error(msg=msg)

	def close(self, close_screens: bool = False):
		''' 
		Function including generic operations when the logger is no more intended to be used,
		such as releasing resources

		In the default version is only close display screen with an option flag.

		:param close_screens: If to close all screens, defaults to False.
		:type close_screens: bool, optional
		'''
		# NOTE: Screens are not closed by default
		#       since they can be managed for other purposes
		#       For this reason we dedicated a specific method `close_all_screens`

		if close_screens:
			self.close_all_screens()

	def set_progress_bar(self):
		''' 
		Logger setup for progress bar
		
		NOTE: Currently only supported for `LoguruLogger` class
		'''

		if isinstance(self, LoguruLogger):

			self._logger.remove()  # Remove default 'stderr' handler
			self._handler = self._logger.add(lambda m: self.CONSOLE.print(m, end=""), colorize=True)

	# --- DIRECTORY ---
	
	def _get_dir(self, conf: Dict[str, str]) -> str:
		'''
		Return the target directory for saving experiment results, 
		building a directory hierarchy across the specified keys:
			- `out_dir`, the result output directory.
			- `title`,   the experiment title.
			- `name`,    the experiment name.
			- `version`, the experiment version, optional.
		
		In the case `version` key is specified it logs at the level of experiment versioning,
		otherwise it logs at the level of the specific experiment name.
		
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

		# If the version is not a specified
		# the target directory is the experiment directory
		if not 'version' in conf:
			return exp_dir

		# If the version key is specified we compute its version
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

			self.warn(msg=f'The new new version will overwrite {target_dir}. Continue [y/n]')
			inp = input()

			# If continue, remove old directory
			if inp == 'y' or inp == 'Y':
				rmdir(target_dir)

			# If not continue, exit
			else:
				self.error("Choose an another version for non overwrite preexisting files. Exiting...")
				exit(1)

		return target_dir
	
	def create_dir(self):
		''' Creates the experiment directory '''
	
		self.info(f"Creating experiment directory {self.dir}")
		os.makedirs(self.dir, exist_ok=True)
		
	@property
	def dir(self) -> str: return self._dir
	''' Returns experiment target directory for saving results. '''

	# --- SCREENS ---

	def add_screen(self, screen: DisplayScreen):
		'''
		Add a new screen with name and size. 
		It raises a key error if that screen name already exists.
		
		:param screen: Display screen object.
		:type screen: DisplayScreen
		'''

		# Check if the screen name already exists
		if str(screen) in self._screens:
			err_msg = f'There already exists a screen with name {screen}.'
			raise KeyError(err_msg)
		
		# Add the screen
		self._screens[str(screen)] = screen

	def update_screen(self, screen_name: str, image: Image.Image):
		'''
		Update a display screen with a new image. It raises a key error if the screen name doesn't exist.
		
		:param screen_name: Name identifier for the new screen name.
		:type screen_name: str.
		:param image: Image to update screen frame.
		:type image: Image.Image
		'''

		# Update the screen
		try:
			self._screens[screen_name].update(image=image)

		# Raise an error if the screen name doesn't exist
		except KeyError:
			raise KeyError(f'Screen {screen_name} not present in screens {self._screens.keys()}.')

	def close_screen(self, screen_name: str):
		'''
		Close a screen by first ending its rendering and then removing it from the list of screens.
		It raises a key error if that screen name doesn't exist.
		
		:param screen_name: Name identifier for the new screen name.
		:type screen_name: str.
		'''

		self.info(f"Closing screen {screen_name}")

		# Stop rendering
		try:
			self._screens[screen_name].close()

		# Raise an error if the screen name doesn't exists
		except KeyError:
			raise KeyError(f'Trying to close screen {screen_name}, but not present in screens {self._screens.keys()}.')

		# Remove from the dictionary
		self._screens.pop(screen_name, None)

	def close_all_screens(self):
		'''
		Close all display screens.
		'''

		# This is necessary to prevent dictionary on-loop changes
		screen_names = list(self._screens.keys())

		# Close all screens
		for screen_name in screen_names:
			self.close_screen(screen_name=screen_name)


	@property
	def screens(self) -> List[str]: return list(self._screens.keys())
	''' Return the name of screens '''

		
class LoguruLogger(Logger):
	''' 
	Logger using `loguru` technology to log both on terminal and file 
	'''

	# NOTE: Loguru technology doesn't provide multiple-logger instances.
	#       For this reason we have a factory-id unique to any instance of
	#       the logger which is bind to the logger object.
	#       This allows to specify a filtering lambda to each file by 
	#       checking the unique logger-id.
	_factory_id = 0

	LOG_FILE = 'info.log'

	def __init__(self, path: Dict[str, str] | str = '.', to_file: bool = True) -> None:
		'''
		Initialize the logger with a possible specific target directory.
		In the case `to_file` flag is active, it logs on file.

		:param path_: Path where to save the experiment results.
			It supports two types of input:
			- Dict[str, str]: dictionary specifiying the target directory hierarchy based
				on experiment directory, title, name and version.
			- str: target directory path.
		:type path_: Dict[str, str] | None
		:param to_file: If to log on file, defaults to True.
		:type to_file: bool, optional
		'''

		# Assign the unique ID
		self._id = self._factory_id
		LoguruLogger._factory_id += 1

		# Initialize logger with unique ID
		self._logger = loguru.logger.bind(id=self._id)

		super().__init__(path=path)

		# File logging
		if to_file:

			# Create the log file and add the handler binding it to the unique ID
			log_file = os.path.join(self.dir, self.LOG_FILE)
			self._handler = self._logger.add(
				log_file, level=0, enqueue=True, 
				filter=lambda x: x['extra']['id'] == self._id
			)
	
	@classmethod
	def from_conf(cls, conf: ParamConfig, to_file: bool = True) -> 'LoguruLogger': 
		'''
		Initialize the logger from parameter configuration
		The parameter requires specifications for output directory and experiment
			- title: it typically refers to the experiment class it is run with, for example `MaximizeActivity`.
			- name: it refers to the specific purpose the experiment is run with.
			- version: it refers to the versioning of the experiment, for example by changing the hyperparameters.

		:param conf: Parameter configuration for the experiment.
		:type conf: ParamConfig
		:param to_file: If to log on file, defaults to True.
		:type to_file: bool, optional
		:return: Mapping for experiment directory, title, name and version.
		:rtype: Dict[str, str]
		'''
		
		try: 
			return LoguruLogger(
				path= {
					'out_dir': str(conf[ArgParams.OutputDirectory  .value]),
					'title'  : str(conf[ArgParams.ExperimentTitle  .value]),
					'name'   : str(conf[ArgParams.ExperimentName   .value]),
					'version': str(conf[ArgParams.ExperimentVersion.value])
				},
				to_file=to_file
			)
		
		except KeyError as e:
			raise KeyError(f'Cannot create Logger from configuration as some keys are missing {e}')
	
	# Overriding logging methods with `loguru` specific ones
	def _info(self, msg: str): self._logger.info   (msg)
	def _warn(self, msg: str): self._logger.warning(msg)
	def _err (self, msg: str): self._logger.error  (msg)

	def close(self, close_screens: bool = False):
		'''
		Close the logger releasing resources. 
		It reimplement superclass `Logger` behavior and removes the logger
		handles to free their references.
		'''

		super().close(close_screens=close_screens)

		if hasattr(self, '_handler'):
			self._logger.remove(handler_id=self._handler)


class SilentLogger(Logger):
	''' Trivial logger with for non-logging '''

	def __init__(self, path: Dict[str, str] | str = '.') -> None:
		''' 
		Initialize the logger
		
		NOTE: `path` input parameter is left for interface compatibility, but ignored
		'''
		super().__init__()
	
	# Override for no logging
	def _info(self, msg: str): pass
	def _warn(self, msg: str): pass
	def _err (self, msg: str): pass


class DisplayScreen:
	''' Screen to display and update images'''

	DEFAULT_DISPLAY_SIZE = (400, 400)

	@staticmethod
	def set_main_screen() -> tk.Tk:
		'''
		Create the main screen for displaying images.
		The main must be created in order to render images in TopLevel additional screens.

		NOTE: The object returned by the function should be saved in a variable
			that mustn't become out of scope; in the case it may be removed by 
			the garbage collector and invalidate screen rendering process.
		'''

		main_screen = tk.Tk()
		main_screen.withdraw()  # hide the main screen

		return main_screen

	def __init__(self, title: str = "Image Display", display_size: Tuple[int, int] = (400, 400)):
		'''
		Initialize a display window with title and size.

		:param title: Screen title, defaults to "Image Display"
		:type title: str, optional
		:param display_size: _description_, defaults to (400, 400)
		:type display_size: tuple, optional
		'''

		# Input parameters
		self._title        = title
		self._display_size = display_size

		# Screen controller
		self._controller = tk.Toplevel()
		self._controller.title(self._title)

		# Create a container frame for the image label
		self._image_frame = tk.Frame(self._controller)
		self._image_label = tk.Label(self._image_frame)

		self._image_frame.pack()
		self._image_label.pack()

	def __str__ (self) -> str: return self._title
	def __repr__(self) -> str: return str(self)


	def update(self, image: Image.Image):
		'''
		Display a new image to the screen

		:param image: New image to be displayed.
		:type image: Image.Image
		'''

		# Resize the image to fit the desired display size
		resized_image = image.resize(self._display_size)

		# Convert the resized image to a Tkinter PhotoImage
		photo = ImageTk.PhotoImage(resized_image)

		# Configure the label with the resized image
		self._image_label.configure(image=photo) # type: ignore
		self._image_label.image = photo          # type: ignore

		# Update the frame
		self._controller.update()

	def close(self):
		'''
		Method to close the screen.

		NOTE: After this, the object becomes useless as the garbage
			collector takes controller with no more reference.
			The object becomes useless and a new one needs to be instantiated.
		'''
		self._controller.after(100, self._controller.destroy)