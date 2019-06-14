# CommonMark changed their interface in 0.8.0. Because older
# Orange is imcompatible with newer CommonMark, 
# install the module manually so that it does not get installed.
# Remove this when the minimum supported Orange is 3.16.
pip install CommonMark==0.7.5

if [ $ORANGE == "release" ]; then
    echo "Orange: Skipping separate Orange install"
    return 0
fi

if [ $ORANGE == "master" ]; then
    echo "Orange: from git master"
    pip install https://github.com/biolab/orange3/archive/master.zip
    return $?;
fi

PACKAGE="orange3==$ORANGE"
echo "Orange: installing version $PACKAGE"
pip install $PACKAGE