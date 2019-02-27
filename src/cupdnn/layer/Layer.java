package cupdnn.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import cupdnn.Network;
import cupdnn.active.ActivationFunc;
import cupdnn.data.Blob;
import cupdnn.data.BlobParams;

public abstract class Layer{
	protected int id;
	protected Network mNetwork;
	protected ActivationFunc activationFunc;
	//BlobParams中的四个参数说明
	//第一个：batch,就是一个批次中有多少个图片
	//第二个：channel,一张图片有多少个通道
	//第三个：图片的高
	//第四个：图片的宽
	public Layer(Network network){
		this.mNetwork = network;
	}
	
	public void setId(int id){
		this.id = id;
	}
	public int getId(){
		return id;
	}
	abstract public Blob createOutBlob();
	abstract public Blob createDiffBlob();
	
	public void setActivationFunc(ActivationFunc func){
		this.activationFunc = func;
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
	
}