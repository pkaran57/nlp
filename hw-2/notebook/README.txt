You could run this Jupyter notebook on Colab or your local machine. 

If you choose to run this Jupyter notebook on your local machine, the following steps might be helpful.

First install conda, close any open terminals you might have, then open a new terminal and run the following:

# 1. Create an environment with dependencies specified in env.yml:
    
    conda env create -f env.yml

# 2. Activate the new environment:
    
    conda activate nlp
    
# 3. Inside the new environment, install IPython kernel so we can use this environment in jupyter notebook: 
    
    python -m ipykernel install --user --name nlp


# 4. With the above done you should be able to get underway by typing:

    jupyter notebook word_vectors.ipynb
    
# 5. To make sure we are using the right environment, go to the toolbar of exploring_word_vectors.ipynb, click on Kernel -> Change kernel, you should see and select nlp in the drop-down menu.

# To deactivate an active environment, use
    
    conda deactivate
