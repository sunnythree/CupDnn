/*
 * 
 * 
 */

package cupcnn.util;
import java.util.Random;

import cupcnn.data.Blob;

public class MathFunctions {
	
	public static void gaussianInitData(double[] data){
		Random random = new Random();
		for(int i=0;i<data.length;i++){
			data[i] = random.nextGaussian()*0.1;
		}
	}
	
	public static void constantInitData(double[] data,double value){
		for(int i=0;i<data.length;i++){
			data[i] = value;
		}
	}
	
	public static void randomInitData(double[] data){
		Random random = new Random();
		for(int i=0;i<data.length;i++){
			data[i] = random.nextDouble();
		}
	}
	
	public static void dataDivConstant(double[] data,double constant){
		for(int i=0; i<data.length;i++){
			data[i] /= constant;
		}
	}
	
	public static void convolutionBlobSame(Blob input,Blob kernel,Blob bias,Blob output){
		double[] inputData = input.getData();
		double[] kernelData = kernel.getData();
		double[] outputData = output.getData();
		double[] biasData = bias.getData();
		
		int features = output.getChannels()/input.getChannels();
		for(int n=0;n<output.getNumbers();n++){
			for(int c=0;c<output.getChannels();c++){
				int inputChannelIndex = c/features;
				for(int h=0;h<output.getHeight();h++){
					for(int w=0;w<output.getWidth();w++){
						//先定位到输出的位置
						//然后遍历kernel,通过kernel定位输入的位置
						//然后将输入乘以kernel
						int inStartX = w - kernel.getWidth()/2;
						int inStartY = h - kernel.getHeight() / 2;
						//和卷积核乘加
						for(int kh=0;kh<kernel.getHeight();kh++){
							for(int kw=0;kw<kernel.getWidth();kw++){
								int inY = inStartY + kh;
								int inX = inStartX + kw;
								if (inY >= 0 && inY < input.getHeight() && inX >= 0 && inX < input.getWidth()){
									outputData[output.getIndexByParams(n,c,h,w)] += kernelData[kernel.getIndexByParams(0,c,kh,kw)]*
											inputData[input.getIndexByParams(n,inputChannelIndex,inY,inX)];
								}
							}
						}
						
						//加偏置
						outputData[output.getIndexByParams(n,c,h,w)] += biasData[bias.getIndexByParams(0, c, 0, 0)];
					}
				}
			}
		}
	}

	public static void convolutionBlobSame(Blob input,Blob kernel,Blob output){
		double[] inputData = input.getData();
		double[] kernelData = kernel.getData();
		double[] outputData = output.getData();
		
		int features = input.getChannels()/output.getChannels();
		for(int n=0;n<input.getNumbers();n++){
			for(int c=0;c<input.getChannels();c++){
				int inputChannelIndex = c/features;
				for(int h=0;h<input.getHeight();h++){
					for(int w=0;w<input.getWidth();w++){
						
						int inStartX = w - kernel.getWidth()/2;
						int inStartY = h - kernel.getHeight() / 2;
						//和卷积核乘加
						for(int kh=0;kh<kernel.getHeight();kh++){
							for(int kw=0;kw<kernel.getWidth();kw++){
								int inY = inStartY + kh;
								int inX = inStartX + kw;
								if (inY >= 0 && inY < output.getHeight() && inX >= 0 && inX < output.getWidth()){
									outputData[output.getIndexByParams(n,inputChannelIndex,inY,inX)] += kernelData[kernel.getIndexByParams(0,c,kh,kw)]*
											inputData[input.getIndexByParams(n,c,h,w)];
								}
							}
						}
					}
				}
			}
		}
	}
	
	public static Blob rotate180Blob(Blob input){
		Blob output = new Blob(input.getNumbers(),input.getChannels(),input.getHeight(),input.getWidth());
		double[] inputData = input.getData();
		double[] outputData = output.getData();
		/*
		 * 旋转180度就是上下颠倒，同时左右镜像
		 */
		for(int n=0;n<output.getNumbers();n++){
			for(int c=0;c<output.getChannels();c++){
				for(int h=0;h<output.getHeight();h++){
					for(int w=0;w<output.getWidth();w++){
						outputData[output.getIndexByParams(n, c, h, w)] = inputData[input.getIndexByParams(n, c, input.getHeight()-h-1, input.getWidth()-w-1)];
					}
				}
			}
		}
		return output;
	}
}
