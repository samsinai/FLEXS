# for AWS EC2 Ubuntu Deep Learning AMI
pip install --upgrade pip
pip install --user --upgrade tf-agents-nightly 
pip install --user --force-reinstall tf-nightly
pip install --user --force-reinstall tfp-nightly
conda install -c bioconda viennarna -y
conda install -c pytorch pytorch 
conda install tqdm 
pip install editdistance
pip install keras
pip install gin
pip install tape_proteins

# then you need to git clone