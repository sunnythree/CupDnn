package test.rnn;

import java.util.List;

import cupcnn.Network;
import cupcnn.active.ReluActivationFunc;
import cupcnn.active.TanhActivationFunc;
import cupcnn.data.Blob;
import cupcnn.layer.FullConnectionLayer;
import cupcnn.layer.InputLayer;
import cupcnn.layer.RecurrentLayer;
import cupcnn.loss.MSELoss;
import cupcnn.optimizer.SGDOptimizer;
import cupcnn.util.DataAndLabel;
import cupcnn.util.DigitImage;

public class AddNetwork {
	Network network;
	SGDOptimizer optimizer;
	
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
		network.setBatch(20);
		network.setLrDecay(0.8f);
		
		network.setLoss(new MSELoss());
		optimizer = new SGDOptimizer(0.8f);
		network.setOptimizer(optimizer);
		
		buildAddNetwork();

		network.prepare();
	}
	public void train(List<DataAndLabel> trainLists,int epoes) {
		network.fit(trainLists, epoes,null);
	}

	public Blob predict(Blob in) {
		return network.predict(in);
	}
	
	public void saveModel(String name){
		network.saveModel(name);
	}
	
	public void loadModel(String name){
		network = new Network();
		network.setBatch(2);
		network.loadModel(name);
		network.prepare();
	}
}
