rm -r model
rm -r summary

for dataset in birds butterfly plant trees vegetation
do
    for num in 1 2 3
    do 
        echo time python train_2.py $dataset $num
        python train_2.py $dataset $num
    done
done


