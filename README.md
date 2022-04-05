# PC-VAIM

**This is the implementation of Point Cloud-based Variational Autoencoder Inverse Mappers (PC-VAIM) - An Application on Quantum Chromodynamics Global Analysis.**


## Requirements
The code is written in Python=3.6, with the following libraries:
* tensorflow==2.8.0
* tensorflow_graphics==2021.12.3
* numpy==1.21.5
See requirements.tex file to get the full list of libraries.

## Getting started
* Install the python libraries. (See [Requirements](https://github.com/ecml-pkdd/PC-VAIM#requirements)).
* Download the code from GitHub:
```bash
git clone https://github.com/ecml-pkdd/PC-VAIM
cd PC-VAIM
```

* Run the python script:
``` bash
python3 train.py
``` 
* This will run the toy example which is f</sub>(x) = ax<sup>2.
* To see a demo using a saved model, go to PC-VAIM_demo.ipynb.
  
  
 ## Results:

 
  ## Example plots results after 500 epochs:
| f</sub>(x) = ax<sup>2| 
|------------|
| <img src="gallery/x2.png" width="250"> |

| latent of  x<sup>2|
|------------|
| <img src="gallery/latent_x2a.png" width="250"> |


