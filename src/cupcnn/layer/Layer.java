package cupcnn.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import cupcnn.data.Blob;
import cupcnn.data.BlobParams;
import cupcnn.Network;
import cupcnn.active.ActivationFunc;

public abstract class Layer{
	protected int id;
	protected Network mNetwork;
	protected ActivationFunc activationFunc;
	protected BlobParams layerParams;
	
	public Layer(Network network,BlobParams parames){
		this.mNetwork = network;
		this.layerParams = parames;
		paramsList = new ArrayList<Blob>();
		gradientList = new ArrayList<Blob>();
	}
	
	public BlobParams getLayerParames(){
		return layerParams;
	}
	
	public void setId(int id){
		this.id = id;
	}
	public int getId(){
		return id;
	}
	public void setActivationFunc(ActivationFunc func){
		this.activationFunc = func;
	}
	public List<Blob> getParamsList(){
		return paramsList;
	}
	
	public List<Blob> getGradientList(){
		return gradientList;
	}
	
	//类型
	abstract public String getType();

	//准备数据
	abstract public void prepare();
	
	//前向传播和反向传播
	abstract public void forward();
	abstract public void backward();
	
	//参数保存和装载
	abstract public void saveModel(ObjectOutputStream out);
	abstract public void loadModel(ObjectInputStream in);
	
	//用来更新参数
	protected List<Blob> paramsList;
	protected List<Blob> gradientList;
}
