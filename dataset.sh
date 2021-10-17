# original source code https://github.com/yunjey/stargan/blob/master/download.sh

URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
ZIP_FILE=./dataset/celeba.zip
mkdir -p ./dataset/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./dataset/
rm $ZIP_FILE