#!/usr/bin/bash

REPO_DIRNAME="csc2529-2022-project"
# Check if repo path exists
echo "Checking if repo exists..."
REPO_PATH=( $(find $HOME -type d -name $REPO_DIRNAME) )

if [[ ${#REPO_PATH[@]} -ge 2 ]]
then
    echo "More than 1 repo directories found!"
    for rpath in "${REPO_PATH[@]}"
    do
        echo -e "\t ${rpath}"
    done
    echo "Please remove unwanted directories!"
    exit 1
fi

echo "Repo found at: ${REPO_PATH}" 

DATA_DIR="${REPO_PATH}/src/data"
if [ ! -d $DATA_DIR ]
then
    echo "No data directory found. Creating data directory..."
    mkdir $DATA_DIR
    echo "Please download all the datasets(training+validation) to the directory: ${DATA_DIR} and run the script again"
    exit
fi

LS_DATA_DIR=( $(ls ${DATA_DIR}) )

if [ ! -d "${DATA_DIR}/MICCAI_BraTS2020" ]
then
    if [[ " ${LS_DATA_DIR[*]} " =~ "MICCAI_BraTS2020_TrainingData.zip" ]] && [[ " ${LS_DATA_DIR[*]} " =~ "MICCAI_BraTS2020_ValidationData.zip" ]]
    then
        echo "Creating root directory for the dataset..."
        mkdir "${DATA_DIR}/MICCAI_BraTS2020"
        echo "Creating train directory for training dataset"
        mkdir "${DATA_DIR}/MICCAI_BraTS2020/train"
        echo "Creating test directory for test dataset"
        mkdir "${DATA_DIR}/MICCAI_BraTS2020/test"

        if [ ! -d "${DATA_DIR}/MICCAI_BraTS2020_TrainingData" ]
        then
            echo "Unzipping training data..."
            unzip -q "${DATA_DIR}/MICCAI_BraTS2020_TrainingData.zip" -d "${DATA_DIR}"
        fi
        if [ ! -d "${DATA_DIR}/MICCAI_BraTS2020_ValidationData" ]
        then    
            echo "Unzipping validation data..."
            unzip -q "${DATA_DIR}/MICCAI_BraTS2020_ValidationData.zip" -d "${DATA_DIR}"
        fi

        echo "Dataset files have been unzipped"
        
        TRAIN_DATA_DIR="${DATA_DIR}/MICCAI_BraTS2020_TrainingData"
        VAL_DATA_DIR="${DATA_DIR}/MICCAI_BraTS2020_ValidationData"

        echo "Reorganizing dataset directory..."
        # Flatten Training data
        find $TRAIN_DATA_DIR -mindepth 2 -type f -exec mv -t $TRAIN_DATA_DIR '{}' +
        find $TRAIN_DATA_DIR -maxdepth 1 -type d -empty -delete -o -type f -empty -delete

        # Flatten Validation data
        find $VAL_DATA_DIR -mindepth 2 -type f -exec mv -t $VAL_DATA_DIR '{}' +
        find $VAL_DATA_DIR -maxdepth 1 -type d -empty -delete -o -type f -empty -delete

        # Organize images for each class
        CLASSES=("flair" "seg" "t1ce" "t1" "t2")
        for class in ${CLASSES[*]}
        do
            mkdir "$TRAIN_DATA_DIR/$class"
            find $TRAIN_DATA_DIR -type f -iname "*${class}.nii.gz" -exec mv -t $TRAIN_DATA_DIR/$class '{}' +
            mkdir "$VAL_DATA_DIR/$class"
            find $VAL_DATA_DIR -type f -iname "*${class}.nii.gz" -exec mv -t $VAL_DATA_DIR/$class '{}' +
        done

        # Move everything under one directory
        echo "Moving stuff to common root directory..."
        mv ${TRAIN_DATA_DIR}/* -t "${DATA_DIR}/MICCAI_BraTS2020/train/"
        mv ${VAL_DATA_DIR}/* -t "${DATA_DIR}/MICCAI_BraTS2020/test/"

        # Remove empty dirs
        echo "Removing empty directories..."
        rm -r ${TRAIN_DATA_DIR}
        rm -r ${VAL_DATA_DIR}
    else
        echo "Please download and place all the datasets(training+validation) to the directory: ${DATA_DIR}"
        exit 1
    fi
else
    echo "Dataset directory is already organized!"
fi

if [ $(dpkg -s tree | grep Status | awk '{print $4}') == "installed" ]
then
    echo "Printing dataset directory structure..." 
    tree -d ${DATA_DIR}
    echo
fi

echo "Done!"