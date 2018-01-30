package test;
 
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class ReadFile {

    private String labelFileName;
    private String imageFileName;

    /** http://yann.lecun.com/exdb/mnist/ **/

    private static final int MAGIC_OFFSET = 0;
    private static final int OFFSET_SIZE = 4; //in bytes

    private static final int LABEL_MAGIC = 2049;
    private static final int IMAGE_MAGIC = 2051;

    private static final int NUMBER_ITEMS_OFFSET = 4;
    private static final int ITEMS_SIZE = 4;

    private static final int NUMBER_OF_ROWS_OFFSET = 8;
    private static final int ROWS_SIZE = 4;
    public static final int ROWS = 28;

    private static final int NUMBER_OF_COLUMNS_OFFSET = 12;
    private static final int COLUMNS_SIZE = 4;
    public static final int COLUMNS = 28;

    private static final int IMAGE_OFFSET = 16;
    private static final int IMAGE_SIZE = ROWS * COLUMNS;
    
    public List<DigitImage> images;


    public ReadFile(String labelFileName, String imageFileName) {
        this.labelFileName = labelFileName;
        this.imageFileName = imageFileName;
    }

    public List<DigitImage> loadDigitImages() throws IOException {
        images = new ArrayList<DigitImage>();

        ByteArrayOutputStream labelBuffer = new ByteArrayOutputStream();
        ByteArrayOutputStream imageBuffer = new ByteArrayOutputStream();

        //  InputStream labelInputStream = this.getClass().getResourceAsStream(labelFileName);
        //  InputStream imageInputStream = this.getClass().getResourceAsStream(imageFileName);
        FileInputStream labelInputStream =new FileInputStream(labelFileName);
        FileInputStream imageInputStream =new FileInputStream(imageFileName);
        
        int read;
        byte[] buffer = new byte[16384];  //16*1024

       
        while((read = labelInputStream.read(buffer, 0, buffer.length)) != -1) {
           labelBuffer.write(buffer, 0, read);
           labelBuffer.flush();
        }

      //  labelBuffer.flush();

        while((read = imageInputStream.read(buffer, 0, buffer.length)) != -1) {
            imageBuffer.write(buffer, 0, read);
            imageBuffer.flush();
        }

       // imageBuffer.flush();

        byte[] labelBytes = labelBuffer.toByteArray();
        byte[] imageBytes = imageBuffer.toByteArray();

        byte[] labelMagic = Arrays.copyOfRange(labelBytes, MAGIC_OFFSET, OFFSET_SIZE);
        byte[] imageMagic = Arrays.copyOfRange(imageBytes, MAGIC_OFFSET, OFFSET_SIZE);

        if(ByteBuffer.wrap(labelMagic).getInt() != LABEL_MAGIC)  {
        	labelInputStream.close();
            imageInputStream.close();
            throw new IOException("Bad magic number in label file!");
        }

        if(ByteBuffer.wrap(imageMagic).getInt() != IMAGE_MAGIC) {
        	labelInputStream.close();
            imageInputStream.close();
            throw new IOException("Bad magic number in image file!");
        }

        int numberOfLabels = ByteBuffer.wrap(Arrays.copyOfRange(labelBytes, NUMBER_ITEMS_OFFSET, NUMBER_ITEMS_OFFSET + ITEMS_SIZE)).getInt();
        int numberOfImages = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_ITEMS_OFFSET, NUMBER_ITEMS_OFFSET + ITEMS_SIZE)).getInt();

        if(numberOfImages != numberOfLabels) {
        	labelInputStream.close();
            imageInputStream.close();
            throw new IOException("The number of labels and images do not match!");
        }

        int numRows = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_OF_ROWS_OFFSET, NUMBER_OF_ROWS_OFFSET + ROWS_SIZE)).getInt();
        int numCols = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_OF_COLUMNS_OFFSET, NUMBER_OF_COLUMNS_OFFSET + COLUMNS_SIZE)).getInt();

        if(numRows != ROWS && numCols != COLUMNS) {
        	labelInputStream.close();
            imageInputStream.close();
            throw new IOException("Bad image. Rows and columns do not equal " + ROWS + "x" + COLUMNS);
        }

        for(int i = 0; i < numberOfLabels; i++) {
            int label = labelBytes[OFFSET_SIZE + ITEMS_SIZE + i];
            byte[] imageData = Arrays.copyOfRange(imageBytes, (i * IMAGE_SIZE) + IMAGE_OFFSET, (i * IMAGE_SIZE) + IMAGE_OFFSET + IMAGE_SIZE);

            images.add(new DigitImage(label, imageData));
        }
        labelInputStream.close();
        imageInputStream.close();
        return images;
    }
}
