package cupcnn.data;

import java.io.Serializable;

public class BlobParams implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private int numbers;
	private int channels;
	private int width;
	private int height;
	
	public BlobParams(int numbers,int channels,int height,int width){
		this.numbers = numbers;
		this.channels = channels;
		this.height = height;
		this.width = width;
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
	
	public void setWidth(int width){
		this.width = width;
	}
	
	public void setHeight(int height){
		this.height = height;
	}
	
	public void setChannels(int channels){
		this.channels = channels;
	}
	
	public void setNumbers(int numbers){
		this.numbers = numbers;
	}
}
