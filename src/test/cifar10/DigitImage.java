package test.cifar10;
public class DigitImage {

	public int label;
	public byte[] imageData;
	
	
	DigitImage(int label, byte[] imageData)
	{
		this.label=label;
		this.imageData=imageData;
	}  

}
