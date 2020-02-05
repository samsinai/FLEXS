# for AWS EC2 Ubuntu Deep Learning AMI
source activate tensorflow2_p36
pip install --upgrade pip
pip install --user --upgrade tf-agents-nightly 
pip install --user --force-reinstall tf-nightly
pip install --user --force-reinstall tfp-nightly
conda install -c bioconda viennarna -y
pip install editdistance
pip install keras
pip install gin

# then you need to git clone