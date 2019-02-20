package test.cifar10;
 
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import cupcnn.util.DigitImage;


public class ReadFile {

    private String fileName;
    private final int picSize = 32*32*3;
    public List<DigitImage> images;


    public ReadFile(String fileName) {
    	this.fileName = fileName;
    }

    public List<DigitImage> loadDigitImages(boolean isTrain) throws IOException {
        images = new ArrayList<DigitImage>();
        if(isTrain) {
	        for(int i=1;i<6;i++) {
	        	String tmpFileName = fileName.replace('%',(char)('0'+i));
	        	//System.out.println(tmpFileName);
	        	FileInputStream fileInputStream =new FileInputStream(tmpFileName);
		        for(int j=0;j<10000;j++) {
		        	int label = fileInputStream.read();
		        	//System.out.println("label: "+label);
		            byte[] buffer = new byte[picSize];  //32*32*3
		        	fileInputStream.read(buffer);
		        	DigitImage digitImage = new DigitImage(label, buffer);
		        	images.add(digitImage);
		        }
		        fileInputStream.close();
		    }
        }else {
        	FileInputStream fileInputStream =new FileInputStream(fileName);
	        for(int j=0;j<10000;j++) {
	        	int label = fileInputStream.read();
	        	//System.out.println("label: "+label);
	            byte[] buffer = new byte[picSize];  //32*32*3
	        	fileInputStream.read(buffer);
	        	DigitImage digitImage = new DigitImage(label, buffer);
	        	images.add(digitImage);
	        }
	        fileInputStream.close();        	
        }
        return images;
    }
}
