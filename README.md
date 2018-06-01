
The Main aim of this project is to alert the Epileptic Patient’s Condition to their family Members to save them if they need help.

Developed Android Application through which patients can able to register, login, and upload their Medical Record(EEG dataset) to the application.
Android application will upload the dataset to the server. After that it will process the data which is returned from the server about a particular patient and If he is in “ictal” or “Pre-ictal” state then using google location services, SMS alert will be sent to his family members and to the nearest Hospital to Rescue him/her.

Implemented several PHP files at the server side to store patients records and to fetch the records from the database.

Dataset Processing is Done Using Python.
•	Used Walsh Hadamard Transform algorithm to construct coefficients from the dataset.
•	Done Bispectrum and BiCoherence analysis using direct FFT method.
•	Calculated Log, Shannon and Norm entropy to classify the patient’s state to three states ( Ictal, pre-Ictal and normal state )

Idea about the project taken from the following Research Paper 
Link to the Research Paper: https://ieeexplore.ieee.org/document/7548991/
