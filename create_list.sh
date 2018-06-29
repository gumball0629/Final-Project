DATA=/home/cos/FP_IML/TrainData/
echo "create train.txt..."
rm -rf train.txt
for i in $(seq 1 50)
do
    for j in $(seq 1 8)
    do
        find $DATA$i -name *_0$j.jpg  >> train.txt
    done
done
echo "Done.."

DATA=/home/cos/FP_IML/TestData/
echo "create test.txt..."
rm -rf test.txt
find $DATA -name *.jpg  >> test.txt
echo "Done.."

DATA=/home/cos/FP_IML/FT_Data/
echo "create FT_test.txt..."
rm -rf FT_test.txt
find $DATA -name *.jpg   >> FT_test.txt
echo "Done.."
