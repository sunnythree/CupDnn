package cupcnn;

import java.util.ArrayList;
import java.util.List;

import cupcnn.data.Blob;
import cupcnn.data.BlobParams;
import cupcnn.layer.InputLayer;
import cupcnn.layer.Layer;
import cupcnn.loss.Loss;
import cupcnn.optimizer.Optimizer;



public class Network {
	private List<Blob> datas;
	private List<Blob> diffs;
	private List<Layer> layers;
	private Loss loss;
	private Optimizer optimizer;
	private int batch;
	
	public Network(){
		datas = new ArrayList<Blob>();
		diffs = new ArrayList<Blob>();
		layers = new ArrayList<Layer>();
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
			BlobParams layerParams = layers.get(i).getLayerParames();
			assert (layerParams.getNumbers()>0 && layerParams.getChannels()>0 && layerParams.getHeight()>0 && layerParams.getWidth() >0):"prapare---layer params error";
			Blob data = new Blob(batch,layerParams.getChannels(),layerParams.getHeight(),layerParams.getWidth());
			datas.add(data);
			Blob diff = new Blob(data.getNumbers(),data.getChannels(),data.getHeight(),data.getWidth());
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
			optimizer.update(layers.get(i).getParamsList(), layers.get(i).getGradientList());
		}
	}
	
	public double  train(Blob inputData,Blob labelData){
		double lossValue = 0.0;
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
	
	public Blob test(Blob inputData){
		Layer first = layers.get(0);
		assert first instanceof InputLayer:"input layer error";
		((InputLayer)first).setInputData(inputData);
	
		//前向传播
		forward();
		//返回最后一层的数据
		return datas.get(datas.size()-1);
	}
	
	public void saveModel(String name){
		
	}
	public void loadModel(String name){
		
	}
}
