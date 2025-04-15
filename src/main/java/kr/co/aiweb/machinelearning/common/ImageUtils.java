package kr.co.aiweb.machinelearning.common;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import ai.djl.Application;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;

/**
 * image utils
 */
public class ImageUtils {
	
	private static final String FILTER_KEY = "backbone";
	private static final String FILTER_VALUE = "mobilenet1.0";

	/**get Criteria<Image, DetectedObjects>
	 * @return
	 */
	private static Criteria<Image, DetectedObjects> getCriteria() {
		Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .optApplication(Application.CV.OBJECT_DETECTION)
                .setTypes(Image.class, DetectedObjects.class)
                .optFilter(FILTER_KEY, FILTER_VALUE) // 또는 yolov5, resnet50 등
                .optProgress(new ProgressBar())
                .build();
		return criteria;
	}
	
	/**이미지에서 탐지가능한 객체명 리스트 취득
	 * @return
	 * @throws Exception
	 */
	public static List<String> detectableClassNamesInImage() throws Exception {
		Criteria<Image, DetectedObjects> criteria = getCriteria();
		try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();) {
			Path p = model.getModelPath();
        	Path c = p.resolve("classes.txt");
        	if(Files.exists(c)) {
        		return Files.readAllLines(c);
        	}
        	return null;
		}
	}
	
	/**이미지에서 탐지한 객체 리스트
	 * @param imgFile
	 * @return
	 * @throws Exception
	 */
	public static List<BufferedImage> detectedObjListInImage(File imgFile) throws Exception {
		List<BufferedImage> boxList = new ArrayList<>();
		Criteria<Image, DetectedObjects> criteria = getCriteria();
        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();
             Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {

        	Image img = ImageFactory.getInstance().fromFile(Paths.get(imgFile.getAbsolutePath()));
            DetectedObjects detections = predictor.predict(img);
            List<DetectedObjects.DetectedObject> items = detections.items();
            //필터링을 해서 객체를 인식할경우 
            //items = items.stream().filter(obj -> obj.getClassName().matches("dog|bird")).collect(Collectors.toList());
            
            // BufferedImage로 변환
            BufferedImage original = (BufferedImage) img.getWrappedImage();

            for (DetectedObjects.DetectedObject obj : items) {
                BoundingBox box = obj.getBoundingBox();
                Rectangle rect = box.getBounds();
                String className = obj.getClassName();
                System.out.println("className = " + className);

                int x = (int) (rect.getX() * original.getWidth());
                int y = (int) (rect.getY() * original.getHeight());
                int w = (int) (rect.getWidth() * original.getWidth());
                int h = (int) (rect.getHeight() * original.getHeight());

                // 박스 잘라내기
                BufferedImage cropped = original.getSubimage(x, y, w, h);
                boxList.add(cropped);
            }
    		return boxList;
        }
	}
	
	/**BufferedImage 를 파일로 저장 (무손실 : png)
	 * @param img
	 * @param saveFile
	 * @throws Exception
	 */
	public static void saveBufferedImg(BufferedImage img, File saveFile) throws Exception {
		ImageIO.write(img, "png", saveFile);
	}
	
	/**
	 * @param inputFile
	 * @param targetWidth
	 * @param targetHeight
	 * @return
	 * @throws IOException
	 */
	public static BufferedImage resizeAndPadImage(File inputFile, int targetWidth, int targetHeight) throws IOException {
        // 원본 이미지 읽기
        BufferedImage originalImage = ImageIO.read(inputFile);
        int originalWidth = originalImage.getWidth();
        int originalHeight = originalImage.getHeight();

        // 원본 비율 기준으로 리사이즈 크기 계산
        double widthRatio = (double) targetWidth / originalWidth;
        double heightRatio = (double) targetHeight / originalHeight;
        double scale = Math.min(widthRatio, heightRatio);

        int newWidth = (int) (originalWidth * scale);
        int newHeight = (int) (originalHeight * scale);

        // 이미지 리사이즈
        java.awt.Image scaledImage = originalImage.getScaledInstance(newWidth, newHeight, java.awt.Image.SCALE_SMOOTH);

        // 새 BufferedImage 생성 (배경 흰색으로)
        BufferedImage outputImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = outputImage.createGraphics();
        g2d.setColor(Color.WHITE); // 배경 색상
        g2d.fillRect(0, 0, targetWidth, targetHeight);

        // 가운데 정렬하여 이미지 배치
        int x = (targetWidth - newWidth) / 2;
        int y = (targetHeight - newHeight) / 2;
        g2d.drawImage(scaledImage, x, y, null);
        g2d.dispose();
        return outputImage;
    }	
	
	
	/**image file filter
	 * @return
	 */
	public static FilenameFilter imgFileFilter() {
		//이미지 파일 filter
		FilenameFilter imageFilter = (dir, name) -> {
            String lowerName = name.toLowerCase();
            return lowerName.endsWith(".jpg") || lowerName.endsWith(".jpeg") ||
                   lowerName.endsWith(".png") || lowerName.endsWith(".gif") ||
                   lowerName.endsWith(".bmp") || lowerName.endsWith(".webp");
        };
        return imageFilter;
	}
	
}
