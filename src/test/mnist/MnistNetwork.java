package test.mnist;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import cupcnn.Network;
import cupcnn.active.ReluActivationFunc;
import cupcnn.active.SigmodActivationFunc;
import cupcnn.active.TanhActivationFunc;
import cupcnn.data.Blob;
import cupcnn.data.BlobParams;
import cupcnn.layer.Conv2dLayer;
import cupcnn.layer.DeepWiseConv2dLayer;
import cupcnn.layer.FullConnectionLayer;
import cupcnn.layer.InputLayer;
import cupcnn.layer.PoolMaxLayer;
import cupcnn.layer.PoolMeanLayer;
import cupcnn.layer.SoftMaxLayer;
import cupcnn.loss.CrossEntropyLoss;
import cupcnn.loss.LogLikeHoodLoss;
import cupcnn.optimizer.Optimizer;
import cupcnn.optimizer.SGDMOptimizer;
import cupcnn.optimizer.SGDOptimizer;

public class MnistNetwork {
	Network network;
	SGDOptimizer optimizer;
	private void buildFcNetwork(){
		//给network添加网络层
		InputLayer layer1 = new InputLayer(network,28,28,1);
		network.addLayer(layer1);
		FullConnectionLayer layer2 = new FullConnectionLayer(network,28*28,100);
		layer2.setActivationFunc(new SigmodActivationFunc());
		network.addLayer(layer2);
		FullConnectionLayer layer3 = new FullConnectionLayer(network,100,30);
		layer3.setActivationFunc(new SigmodActivationFunc());
		network.addLayer(layer3);
		FullConnectionLayer layer4 = new FullConnectionLayer(network,30,10);
		layer4.setActivationFunc(new SigmodActivationFunc());
		network.addLayer(layer4);
	}
	
	private void buildConvNetwork(){
		InputLayer layer1 =  new InputLayer(network,28,28,1);
		network.addLayer(layer1);
		
		Conv2dLayer conv1 = new Conv2dLayer(network,28,28,1,6,5,1);
		conv1.setActivationFunc(new ReluActivationFunc());
		network.addLayer(conv1);
		
		PoolMaxLayer pool1 = new PoolMaxLayer(network,28,28,6,2,2);
		network.addLayer(pool1);
		
		Conv2dLayer conv2 = new Conv2dLayer(network,14,14,6,36,3,1);
		conv2.setActivationFunc(new ReluActivationFunc());
		network.addLayer(conv2);
		
		PoolMeanLayer pool2 = new PoolMeanLayer(network,14,14,36,2,2);
		network.addLayer(pool2);
		
		
		FullConnectionLayer fc1 = new FullConnectionLayer(network,7*7*36,500);
		fc1.setActivationFunc(new ReluActivationFunc());
		network.addLayer(fc1);
		
		FullConnectionLayer fc2 = new FullConnectionLayer(network,500,50);
		fc2.setActivationFunc(new ReluActivationFunc());
		network.addLayer(fc2);
		
		FullConnectionLayer fc3 = new FullConnectionLayer(network,50,10);
		fc3.setActivationFunc(new SigmodActivationFunc());
		network.addLayer(fc3);
		
	}
	public void buildNetwork(int numOfTrainData){
		//首先构建神经网络对象，并设置参数
		network = new Network();
		network.setThreadNum(8);
		network.setBatch(50);
		//network.setLoss(new LogLikeHoodLoss());
		network.setLoss(new CrossEntropyLoss());
		optimizer = new SGDOptimizer(1f);
		network.setOptimizer(optimizer);
		
		buildFcNetwork();
		//buildConvNetwork();

		network.prepare();
	}
	
	public List<Blob> buildBlobByImageList(List<DigitImage> imageList,int start,int batch,int channel,int height,int width){
		Blob input = new Blob(batch,channel,height,width);
		Blob label = new Blob(batch,network.getDatas().get(network.getDatas().size()-1).getWidth());
		label.fillValue(0);
		float[] blobData = input.getData();
		float[] labelData = label.getData();
		for(int i=start;i<(batch+start);i++){
			DigitImage img = imageList.get(i);
			byte[] imgData = img.imageData;
			assert img.imageData.length== input.get3DSize():"buildBlobByImageList -- blob size error";
			for(int j=0;j<imgData.length;j++){
				blobData[(i-start)*input.get3DSize()+j] = (imgData[j]&0xff)/128.0f-1;//normalize and centerlize(-1,1)
			}
			int labelValue = img.label;
			for(int j=0;j<label.getWidth();j++){
				if(j==labelValue){
					labelData[(i-start)*label.getWidth()+j] = 1;
				}
			}
		}
		List<Blob> inputAndLabel = new ArrayList<Blob>();
		inputAndLabel.add(input);
		inputAndLabel.add(label);
		return inputAndLabel;
	}
	
	private int getMaxIndexInArray(double[] data){
		int maxIndex = 0;
		double maxValue = 0;
		for(int i=0;i<data.length;i++){
			if(maxValue<data[i]){
				maxValue = data[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	private int[] getBatchOutputLabel(float[] data){
		int[] outLabels = new int[network.getDatas().get(network.getDatas().size()-1).getHeight()];
		int outDataSize = network.getDatas().get(network.getDatas().size()-1).getWidth();
		for(int n=0;n<outLabels.length;n++){
			int maxIndex = 0;
			double maxValue = 0;
			for(int i=0;i<outDataSize;i++){
				if(maxValue<data[n*outDataSize+i]){
					maxValue = data[n*outDataSize+i];
					maxIndex = i;
				}	
			}
			outLabels[n] = maxIndex;
		}
		return outLabels;
	}
	
	private void testInner(Blob input,Blob label){
		Blob output = network.predict(input);
		int[] calOutLabels = getBatchOutputLabel(output.getData());
		int[] realLabels = getBatchOutputLabel(label.getData());
		assert calOutLabels.length == realLabels.length:"network train---calOutLabels.length == realLabels.length error";
		int correctCount = 0;
		for(int kk=0;kk<calOutLabels.length;kk++){
			if(calOutLabels[kk] == realLabels[kk]){
				correctCount++;
			}
		}
		double accuracy = correctCount/(1.0*realLabels.length);
		System.out.println("accuracy is "+accuracy);
	}
	
	
	public void train(List<DigitImage> trainLists,int epoes,List<DigitImage> testLists){
		System.out.println("training...... please wait for a moment!");
		int batch = network.getBatch();
		float loclaLr = optimizer.getLr();
		float lossValue = 0;
		
		for(int e=0;e<epoes;e++){
			Collections.shuffle(trainLists);
			long start = System.currentTimeMillis();
			for(int i=0;i<=trainLists.size()-batch;i+=batch){
				List<Blob> inputAndLabel = buildBlobByImageList(trainLists,i,batch,1,28,28);
				float tmpLoss = network.train(inputAndLabel.get(0), inputAndLabel.get(1));
				lossValue = (lossValue+tmpLoss)/2;
				if(i%1000==0) {
					System.out.print(".");
				}
			}
			//每个epoe做一次测试
			System.out.println();
			System.out.println("training...... epoe: "+e+" lossValue: "+lossValue
					+"  "+" lr: "+optimizer.getLr()+"  "+" cost "+(System.currentTimeMillis()-start));
		
			test(testLists);
			
			if(loclaLr>0.00001f){
				loclaLr*=0.8f;
				optimizer.setLr(loclaLr);
			}
		}
	}
	

	
	public void test(List<DigitImage> imgList){
		System.out.println("testing...... please wait for a moment!");
		int batch = network.getBatch();
		int correctCount = 0;
		int allCount = 0;
		int i = 0;
		for(i=0;i<=imgList.size()-batch;i+=batch){
			allCount += batch;
			List<Blob> inputAndLabel = buildBlobByImageList(imgList,i,batch,1,28,28);
			Blob output = network.predict(inputAndLabel.get(0));
			int[] calOutLabels = getBatchOutputLabel(output.getData());
			int[] realLabels = getBatchOutputLabel(inputAndLabel.get(1).getData());
			for(int kk=0;kk<calOutLabels.length;kk++){
				if(calOutLabels[kk] == realLabels[kk]){
					correctCount++;
				}
			}
		}
		
		float accuracy = correctCount/(float)allCount;
		System.out.println("test accuracy is "+accuracy+" correctCount "+correctCount+" allCount "+allCount);
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
