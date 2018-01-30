package cupcnn.data;

public class BlobParams {
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
}
