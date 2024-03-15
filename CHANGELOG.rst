*********
Changelog
*********

All important and notable changes to the megamusicaps projects

0.0.7 (2024-03-15)
==================

⚡️ Changed
-----------

* caption_generator.py: Added examples of output to prompt to improve generated captions from GPT
* In btc chord extractor configs, reduce inst len to 9 as some audio files can be slightly lesser than 10 seconds (musicaps) .This will prevent chord extraction failures for shorter files.
* In main process, add tqdm progress bar for visualisation
* feature_extractor/btc_chord_extractor: Remove redundant prints and logging
* feature_extractors/key_classification: Add key classification extractor class,inference code and models
* main.py: Integrate key classification with main pipeline
* caption_generator_config.yaml: Add key classifier configs to caption generator configs

🛠️ Fixed
---------

* Catch any exceptions during a single caption generation error and continue for the rest of the paths


0.0.6 (2024-01-27)
==================

✨ Added
---------

* Add simple gui to listen to snippet and read audio
* Add script to create json file required by pre-processing script, which takes in directories and puts all mp3 files into the output json
* Add fix for beatnet extractor to prevent post processing if no beats are generated
* Add samples, their post-processed snippets and their generated captions for ease of reviewing performance

Contributor(s): annabeth97c


0.0.5 (2024-01-26)
==================
 
✨ Added
---------

* Add source separation code in feature extractors directory
* Create preprocessing script to allow splitting original audio into 30 second segments
* Add source separation function to preprocessing step for each 30 second segment
* Create new input json with paths to new 30 second segment files

⚡️ Changed
-----------

* Modified main loop to allow choosing source from "raw", "vocals", "drums", "other", "bass" for each extractor, to allow for better tag extraction
* Configured beat extractor to use "drums"
* Configured gender extractor to use "vocals"
* Change gender extractor to classify based on threshold. If neutral, outputs inconclusive
 
🛠️ Fixed
---------

* Add fix to beat extractor to check for too few peaks before generating repeated pattern to prevent crash

Contributor(s): annabeth97c


0.0.4 (2023-12-25)
==================
 
✨ Added
---------

* Add gender classification by voice inference module
* Add gender extractor class that inherits from feuture extractor
* Add naive post-processing to output tag as "male" or "female"

⚡️ Changed
-----------

* Deactivate essentia gender extractor in config file
 
🛠️ Fixed
---------

* Maintain correct size of extractors list by appending None to the extractor list when extractor is disabled. Previously toggling an extractor to be disabled caused a crash

Contributor(s): annabeth97c


0.0.3 (2023-12-24)
==================
 
✨ Added
---------

* Add chord extraction inference scripts and utils
* Create chord extractor class inheriting from feature extractor base class
   
⚡️ Changed
-----------

* Update prompt to ask for a more summarised response regarding chords 

Contributor(s): annabeth97c


0.0.2 (2023-12-20)
==================
 
✨ Added
---------

* Add beatnet extractor inheriting from feature extractor base class
* Add beat detection inference module
* Add naive post processing of beat to get bpm, rhythm and repeated pattern
   
⚡️ Changed
-----------

* Load audio within feature extractor instead of in the main process to allow different ways of loading
 
🛠️ Fixed
---------

* Change output of get_tags in essentia extractor to be a list 

Contributor(s): annabeth97c

 
0.0.1 (2023-12-15)
==================
 
✨ Added
---------

* Create modular pipeline in main.py for:
  * loading audio
  * calling each feature extractor to extract tags
  * converting extracted tags to a chat gpt prompt
  * storing the tags as well as generated caption in json file
* Create base class for feature extractors
* Create child classes that inherit from the feature extractor base class to implement:
  * essentia tag extraction
  * essentia voice tag extraction
* Add conda environment yaml for ease of set up
* Add readme for better documentation
   
⚡️ Changed
-----------

* Moved audio preprocessing scripts to utility directory

🗑️ Removed
-----------

* Original essentia scripts

Contributor(s): annabeth97c


0.0.0 (2023-11-21)
==================
 
✨ Added
---------

* Essentia tag extraction system
* Simple preprocessing script for splitting

Contributor(s): Dapwner
