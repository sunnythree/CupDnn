# CupCnn
A Java implement of Convolutional Neural Network. 


## Build a CNN
```
	private void buildConvNetwork(){
		InputLayer layer1 = new InputLayer(network,new BlobParams(network.getBatch(),1,28,28));
		network.addLayer(layer1);
		
		ConvolutionLayer conv1 = new ConvolutionLayer(network,new BlobParams(network.getBatch(),6,28,28),new BlobParams(1,6,3,3));
		conv1.setActivationFunc(new ReluActivationFunc());
		network.addLayer(conv1);
		
		PoolMaxLayer pool1 = new PoolMaxLayer(network,new BlobParams(network.getBatch(),6,14,14),new BlobParams(1,6,2,2),2,2);
		network.addLayer(pool1);
		
		ConvolutionLayer conv2 = new ConvolutionLayer(network,new BlobParams(network.getBatch(),12,14,14),new BlobParams(1,12,3,3));
		conv2.setActivationFunc(new ReluActivationFunc());
		network.addLayer(conv2);
		
		PoolMaxLayer pool2 = new PoolMaxLayer(network,new BlobParams(network.getBatch(),12,7,7),new BlobParams(1,12,2,2),2,2);
		network.addLayer(pool2);
		
		FullConnectionLayer fc1 = new FullConnectionLayer(network,new BlobParams(network.getBatch(),512,1,1));
		fc1.setActivationFunc(new ReluActivationFunc());
		network.addLayer(fc1);
		
		FullConnectionLayer fc2 = new FullConnectionLayer(network,new BlobParams(network.getBatch(),64,1,1));
		fc2.setActivationFunc(new ReluActivationFunc());
		network.addLayer(fc2);
		
		FullConnectionLayer fc3 = new FullConnectionLayer(network,new BlobParams(network.getBatch(),10,1,1));
		fc3.setActivationFunc(new ReluActivationFunc());
		network.addLayer(fc3);
		
		SoftMaxLayer sflayer = new SoftMaxLayer(network,new BlobParams(network.getBatch(),10,1,1));
		network.addLayer(sflayer);
		
	}
```

	
## Pull Request
Pull request is welcome.

## communicate with
QQ group£º704153141  

## Features
1.without any dependency
2.Basic layer: input layer, convolution layer, pooling layer, full connect layer, softmax layer
3.Loss function: Cross Entropy,log like-hood
4.Optimize method: SGD
5.active funcs:sigmod , tanh, relu

##License
BSD 2-Clause
	
			


