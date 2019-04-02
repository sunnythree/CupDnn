# CupDnn
A Java implement of Deep Neural Network. 


## Build a CNN Network
```
	public void buildNetwork(int numOfTrainData){
		//首先构建神经网络对象，并设置参数
		network = new Network();
		network.setThreadNum(8);
		network.setBatch(20);
		network.setLrAttenuation(0.9f);
		network.setLoss(new MSELoss());
		optimizer = new SGDOptimizer(0.1f);
		network.setOptimizer(optimizer);
	
		buildConvNetwork();

		network.prepare();
	}
	
	private void buildConvNetwork(){
		InputLayer layer1 =  new InputLayer(network,28,28,1);
		network.addLayer(layer1);
		
		Conv2dLayer conv1 = new Conv2dLayer(network,28,28,1,8,3,1);
		conv1.setActivationFunc(new ReluActivationFunc());
		network.addLayer(conv1);
		
		PoolMaxLayer pool1 = new PoolMaxLayer(network,28,28,8,2,2);
		network.addLayer(pool1);
		
		Conv2dLayer conv2 = new Conv2dLayer(network,14,14,8,8,3,1);
		conv2.setActivationFunc(new ReluActivationFunc());
		network.addLayer(conv2);
	
		PoolMeanLayer pool2 = new PoolMeanLayer(network,14,14,8,2,2);
		network.addLayer(pool2);
	
		FullConnectionLayer fc1 = new FullConnectionLayer(network,7*7*8,256);
		fc1.setActivationFunc(new ReluActivationFunc());
		network.addLayer(fc1);
		
		FullConnectionLayer fc2 = new FullConnectionLayer(network,256,10);
		fc2.setActivationFunc(new ReluActivationFunc());
		network.addLayer(fc2);
		
		SoftMaxLayer sflayer = new SoftMaxLayer(network,10);
		network.addLayer(sflayer);
		
	}
```



## Build a RNN Network
```
public void buildAddNetwork() {
		InputLayer layer1 =  new InputLayer(network,2,1,1);
		network.addLayer(layer1);
		RecurrentLayer rl = new RecurrentLayer(network,RecurrentLayer.RecurrentType.RNN,2,2,100);
		network.addLayer(rl);
		FullConnectionLayer fc = new FullConnectionLayer(network,100,2);
		network.addLayer(fc);
	}
	public void buildNetwork(){
		//首先构建神经网络对象，并设置参数
		network = new Network();
		network.setThreadNum(4);
		network.setBatch(100);
		network.setLrDecay(0.7f);
		
		network.setLoss(new MSELoss());//CrossEntropyLoss
		optimizer = new SGDOptimizer(0.9f);
		network.setOptimizer(optimizer);
		
		buildAddNetwork();

		network.prepare();
	}
```
	
## Pull Request
Pull request is welcome.

## communicate with
QQ group: 704153141  

## Features
1.without any dependency<br />
2.Basic layer: input layer, conv2d layer,deepwise conv2d layer, pooling layer(MAX and MEAN), full connect layer, softmax layer, recurrent layer <br />
3.Loss function: Cross Entropy,log like-hood ,MSE loss<br />
4.Optimize method: SGD(SGD without momentum),SGDM(SGD with momentum)<br />
5.active funcs:sigmod , tanh, relu<br />
6.L1 and L2 regularization is supported.<br />
7.Support for multi-threaded acceleration<br />

## Test
mnist test is offered(2017).<br />
cifar10 test is offered(2018-12-23).

## Performance
Can achieve 99% accuracy in mnist dataset(10 conv2d + pool max + 10 conv2d + pool mean + 256 fc + 10 fc + softmax).


##License
BSD 3-Clause
	
			


