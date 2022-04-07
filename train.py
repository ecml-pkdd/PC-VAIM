if __name__ == "__main__":
  
    # -- set up the gpu env
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # -- import the model
    from pc_vaim import *

    # -- generate toy data
    X_train, X_test, y_train, y_test = generate_data()

    # -- instantiate pcvaim object
    pcvaim = PCVAIM()

    # -- train the model
    pcvaim.train(X_train, y_train)

    # -- get latent variables and sample
    latent_mean, latent_std, z = predict(pcvaim, y_train)
    z_samples = sample(latent_mean, latent_std, y_train)
    point_cloud = generate_test_example(z)
    results = pcvaim.decoder.predict([z_samples, point_cloud])

    # plot results on test example
    plt.figure(figsize = (10,4))
    plt.tick_params(labelsize = 20)
    a = results[0]
    plt.hist(a[:,0], bins=100, histtype = 'step', color = 'darkgreen', label = r'$\rm PC-VAIM$')
    plt.xlabel(r'$\rm a $', size=20)
    plt.axvline(1, linestyle='dashed',color= 'r',label=r'$\rm  True$' )
    plt.axvline(-1, linestyle='dashed',color= 'r')
    plt.legend(frameon = 0, loc = 'upper center', fontsize = 20)
    plt.savefig('gallery/results.png')
