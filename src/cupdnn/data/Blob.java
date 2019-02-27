package cupdnn.data;
/*
 *cupcnn的核心数据类
 */

import java.io.Serializable;

public class Blob implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private float[] data;
	private int numbers;
	private int channels;
	private int width;
	private int height;
	private int id;
	private int dim;
	
	public Blob(Blob b,boolean copy) {
		this.numbers = b.numbers;
		this.channels = b.channels;
		this.height = b.height;
		this.width = b.width;
		this.dim = b.dim;
		data = new float[getSize()];	
		if(copy) {
			for(int i=0;i<data.length;i++) {
				data[i]=b.getData()[i];
			}
		}else {
			for(int i=0;i<data.length;i++) {
				data[i]=0;
			}
		}
	}
	
	public Blob(int width){
		this.width = width;
		this.dim = 1;
		data = new float[width];
	}
	
	public Blob(int height,int width){
		this.height = height;
		this.width = width;
		this.dim = 2;
		data = new float[get2DSize()];
	}
	
	public Blob(int channels,int height,int width){
		this.channels = channels;
		this.height = height;
		this.width = width;
		this.dim = 3;
		data = new float[get3DSize()];
	}
	
	public Blob(int numbers,int channels,int height,int width){
		this.numbers = numbers;
		this.channels = channels;
		this.height = height;
		this.width = width;
		this.dim = 4;
		data = new float[getSize()];
	}

	
	//获取第n个number的第channels个通道的第height行的第width列的数
	public double getDataByParams(int numbers,int channels,int height,int width){
		return data[numbers*get3DSize()+channels*get2DSize()+height*getWidth()+width];
	}
	
	public int getIndexByParams(int numbers,int channels,int height,int width){
		return (numbers*get3DSize()+channels*get2DSize()+height*getWidth()+width);
	}
	
	public int getWidth(){
		return width;
	}
	
	public int getHeight(){
		return height;
	}
	
	public int getChannels(){
		return channels;
	}
	
	public int getNumbers(){
		return numbers;
	}
	
	public int get2DSize(){
		return width * height;
	}
	
	public int get3DSize(){
		return channels*width*height;
	}
	
	public int get4DSize(){
		return numbers*channels*width*height;
	}
	
	public int getSize(){
		if(dim==1) {
			return width;
		}else if(dim==2) {
			return get2DSize();
		}else if(dim==3) {
			return get3DSize();
		}else {
			return get4DSize();
		}
	}
	
	public void setId(int id){
		this.id = id;
	}
	
	public int getId(){
		return id;
	}
	
	public float[] getData(){
		return data;
	}
	
	public void fillValue(float value){
		for(int i=0;i<data.length;i++){
			data[i] = value;
		}
	}
	
	public void cloneTo(Blob to){
		to.numbers = this.numbers;
		to.channels = this.channels;
		to.height = this.height;
		to.width = this.width;
		float[] toData = to.getData();
		for(int i=0;i<data.length;i++){
			toData[i] = this.data[i];
		}
	}
	
}
