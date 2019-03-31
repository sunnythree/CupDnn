package test.rnn;

import java.util.List;

import cupcnn.Network;
import cupcnn.active.ReluActivationFunc;
import cupcnn.data.Blob;
import cupcnn.layer.FullConnectionLayer;
import cupcnn.layer.RecurrentLayer;
import cupcnn.loss.MSELoss;
import cupcnn.optimizer.SGDOptimizer;
import cupcnn.util.DataAndLabel;
import cupcnn.util.DigitImage;

public class AddNetwork {
	Network network;
	SGDOptimizer optimizer;
	
	public void buildAddNetwork() {
		RecurrentLayer rl = new RecurrentLayer(network,RecurrentLayer.RecurrentType.RNN,2,1,10);
		network.addLayer(rl);
		FullConnectionLayer fc = new FullConnectionLayer(network,10,1);
		fc.setActivationFunc(new ReluActivationFunc());
		network.addLayer(fc);
	}
	public void buildNetwork(int numOfTrainData){
		//首先构建神经网络对象，并设置参数
		network = new Network();
		network.setThreadNum(4);
		network.setBatch(20);
		network.setLrAttenuation(0.9f);
		
		network.setLoss(new MSELoss());
		optimizer = new SGDOptimizer(0.1f);
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
		network.loadModel(name);
		network.prepare();
	}
}
