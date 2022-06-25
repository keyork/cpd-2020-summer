# Download dataset

if [ ! -d './data/' ];then
    mkdir ./data
    echo 'Data floder not exists, create: ./data/'
else
    echo 'Data floder already exists: ./data/'
fi

cd data

if [ ! -d './lfw/' ]; then
    echo 'RAW Data floder not exists'
    if [ ! -f './lfw.tgz' ]; then
        echo 'RAW Data ompression file not exists'
        echo 'Download'
        wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
        tar -zxvf lfw.tgz
    else
        echo 'RAW Data floder already exists: ./data/lfw.tgz'
        echo 'Unpack'
        tar -zxvf lfw.tgz
    fi
else
    echo 'RAW Data folder already exists: ./data/lfw/'
fi

# Process Data
# Split by Gender

echo 'Process Data'

if [ ! -d './processed/' ];then
    cp -r ../label/ ./
    python ../utils/process.py
    rm -rf ./label
else
    echo 'Dataset already Processed: ./processed/'
fi

echo 'Done'