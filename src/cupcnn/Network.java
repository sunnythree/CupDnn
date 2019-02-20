package cupcnn;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import cupcnn.data.Blob;
import cupcnn.data.BlobParams;
import cupcnn.layer.Conv2dLayer;
import cupcnn.layer.DeepWiseConv2dLayer;
import cupcnn.layer.FullConnectionLayer;
import cupcnn.layer.InputLayer;
import cupcnn.layer.Layer;
import cupcnn.layer.PoolMaxLayer;
import cupcnn.layer.PoolMeanLayer;
import cupcnn.layer.SoftMaxLayer;
import cupcnn.loss.Loss;
import cupcnn.optimizer.Optimizer;



public class Network{
	public static String MODEL_BEGIN = "BEGIN";
	public static String MODEL_END = "END";
	private List<Blob> datas;
	private List<Blob> diffs;
	private List<Layer> layers;
	private Loss loss;
	private Optimizer optimizer;
	private int batch = 1;
	private int threadNum = 4;
	
	public Network(){
		datas = new ArrayList<Blob>();
		diffs = new ArrayList<Blob>();
		layers = new ArrayList<Layer>();
	}
	
	public int getThreadNum() {
		return threadNum;
	}
	
	public void setThreadNum(int num) {
		threadNum = num;
	}
	/*
	 *添加创建的层
	 */
	public void addLayer(Layer layer){
		layers.add(layer);
	}
	
	/*
	 * 获取datas
	 */
	public List<Blob> getDatas(){
		return datas;
	}
	/*
	 * 获取diffs
	 */
	public List<Blob> getDiffs(){
		return diffs;
	}
	/*
	 * 获取Layers
	 */
	public List<Layer> getLayers(){
		return layers;
	}

	
	public void setLoss(Loss loss){
		this.loss = loss;
	}

	
	public void setBatch(int batch){
		this.batch = batch;
	}
	
	public int getBatch(){
		return this.batch;
	}
	
	public void setOptimizer(Optimizer optimizer){
		this.optimizer = optimizer;
	}
	
	public void prepare(){
		for(int i=0;i<layers.size();i++){
			Blob data = layers.get(i).createOutBlob();
			datas.add(data);
			Blob diff = layers.get(i).createDiffBlob();
			diffs.add(diff);
			layers.get(i).setId(i);
			layers.get(i).prepare();
		}
	}
	
	
	public void forward(){
		for(int i=0;i<layers.size();i++){
			layers.get(i).forward();
		}
	}
	

	
	public void backward(){

		for(int i=layers.size()-1;i>-1;i--){
			layers.get(i).backward();
			//使用优化器更新参数
			optimizer.updateW(layers.get(i).getParamsWList(), layers.get(i).getGradientWList());
			optimizer.updateB(layers.get(i).getParamsBList(), layers.get(i).getGradientBList());
		}
	}
	
	public float  train(Blob inputData,Blob labelData){
		float lossValue = 0.0f;
		Layer first = layers.get(0);
		assert first instanceof InputLayer:"input layer error";
		((InputLayer)first).setInputData(inputData);
	
		//前向传播
		forward();

		
		//计算输出误差
		lossValue = loss.loss(labelData, datas.get(datas.size()-1));
		//计算输出层diff
		loss.diff(labelData, datas.get(datas.size()-1), diffs.get(diffs.size()-1));
		

		
		//反响传播
		backward();
		
		return lossValue;
	}
	
	public Blob predict(Blob inputData){
		Layer first = layers.get(0);
		assert first instanceof InputLayer:"input layer error";
		((InputLayer)first).setInputData(inputData);
	
		//前向传播
		forward();
		//返回最后一层的数据
		return datas.get(datas.size()-1);
	}
	
	public void saveModel(String name){
		System.out.println("begin save model");
		ObjectOutputStream out = null;
	    try {
			out = new ObjectOutputStream(new FileOutputStream(name));
			out.writeUTF(MODEL_BEGIN);
			out.writeInt(layers.size());
			for(int i=0;i<layers.size();i++){
				layers.get(i).saveModel(out);
			}
			out.writeUTF(MODEL_END);
			out.flush();
			out.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    try {
			out.flush();
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    System.out.println("save model finished");
	}
	
	public void loadModel(String name){
		System.out.println("begin load model");
		ObjectInputStream in = null;
		try {
			in = new ObjectInputStream(new FileInputStream(name));
			String begin = in.readUTF();
			if(!begin.equals(MODEL_BEGIN)){
				System.out.println("file format error");
				in.close();
				return;
			}
			int layersSize = in.readInt();
			if(layersSize<=0){
				System.out.println("no layers");
				in.close();
				return;			
			}
			String layerType = null;
			for(int i=0;i<layersSize;i++){
				layerType = in.readUTF();
				if(layerType.equals(InputLayer.TYPE)){
					InputLayer inputLayer = new InputLayer(Network.this);
					inputLayer.loadModel(in);
					layers.add(inputLayer);
				}else if(layerType.equals(DeepWiseConv2dLayer.TYPE)){
					DeepWiseConv2dLayer conv = new DeepWiseConv2dLayer(Network.this);
					conv.loadModel(in);
					layers.add(conv);
				}else if(layerType.equals(Conv2dLayer.TYPE)){
					Conv2dLayer conv = new Conv2dLayer(Network.this);
					conv.loadModel(in);
					layers.add(conv);
				}else if(layerType.equals(FullConnectionLayer.TYPE)){
					FullConnectionLayer fc = new FullConnectionLayer(Network.this);
					fc.loadModel(in);
					layers.add(fc);
				}else if(layerType.equals(PoolMaxLayer.TYPE)){
					PoolMaxLayer pMax = new PoolMaxLayer(Network.this);
					pMax.loadModel(in);
					layers.add(pMax);
				}else if(layerType.equals(PoolMeanLayer.TYPE)){
					PoolMeanLayer pMean = new PoolMeanLayer(Network.this);
					pMean.loadModel(in);
					layers.add(pMean);
				}else if(layerType.equals(SoftMaxLayer.TYPE)){
					SoftMaxLayer softMax = new SoftMaxLayer(Network.this);
					softMax.loadModel(in);
					layers.add(softMax);
				}else{
					System.out.println("load model error");
					System.exit(-1);
				}
			}
			String end = in.readUTF();
			if(!end.equals(MODEL_END)){
				System.out.println("end is "+end+" file format error");
			}
			in.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("load model finished");
	}
}
