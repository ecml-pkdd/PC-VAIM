# PC-VAIM

**This is the implementation of Point Cloud-based Variational Autoencoder Inverse Mappers (PC-VAIM) - An Application on Quantum Chromodynamics Global Analysis.**


## Requirements
The code is written in Python=3.6, with the following libraries:
* tensorflow==2.8.0
* tensorflow_graphics==2021.12.3
* numpy==1.21.5
-------------------------------------------------------------------
* To get the list of the libraries, see the file:
``` bash
requirements.txt
``` 
* To get the list of the libraries with all of the dependencies, see the file:
``` bash
requirements_dependency.txt
``` 

## Getting started
* Install the python libraries.
* Download the code from GitHub:
```bash
git clone https://github.com/ecml-pkdd/PC-VAIM.git
cd PC-VAIM
```

* Run the python script:
``` bash
python3 train.py
``` 
* This will run the toy example f</sub>(x) = (ax)<sup>2.
* To see a demo using a saved model, go to PC-VAIM_demo.ipynb.
  
  
 ## Results:

 ## Example on toy data after 500 epochs:
| f</sub>(x) = ax<sup>2      | latent      |
|------------|-------------|
| <img src="gallery/result.png" width="350"> | <img src="gallery/latent.png" width="350"> |

