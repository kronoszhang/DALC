# the trained model only used for 4x UpSampling and must delete ./output folder when run code
# the data struct is flooowing:

#data
#    Market1501
#          bounding_box_train/
#          bounding_box_test/
#          ......


python test_market.py --dataset 'folder' --dataroot './data' --batchSize 4  --upSampling 4 --cuda --generatorWeights 'checkpoints/generator_final.pth' --discriminatorWeights  'checkpoints/discriminator_final.pth'