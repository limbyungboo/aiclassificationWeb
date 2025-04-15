/**
 * 
 */
package kr.co.aiweb.machinelearning.trainmodel;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.factory.Nd4j;

import kr.co.aiweb.machinelearning.common.ImageUtils;
import kr.co.aiweb.machinelearning.common.MLConst.DatasetConst;
import kr.co.aiweb.machinelearning.vo.LabelInfo;
import kr.co.aiweb.machinelearning.vo.PredictResult;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

/**
 * 
 */
@Slf4j
public abstract class TrainModel {
	
	/**
	 * image trainformer
	 */
	protected DataNormalization preProcessor;

	/**
	 * machine learning root directory
	 */
	@Getter
	protected File rootDir;
	/**
	 * model file
	 */
	protected File modelFile;
	/**
	 * label file
	 */
	protected File labelFile;
	
	/**
	 * dataset directory
	 */
	@Getter
	protected File datasetDir;
	
	/**
	 * label 정보
	 */
	@Getter
	protected LabelInfo labelInfo;
	
	/**
	 * image loader
	 */
	protected NativeImageLoader loader = new NativeImageLoader(DatasetConst.HEIGHT.getValue(), DatasetConst.WIDTH.getValue(), DatasetConst.CHANNELS.getValue());
	
	
	/**constructor
	 * @param rootDir
	 * @throws Exception
	 */
	protected TrainModel(File rootDir) throws Exception {
		this.rootDir = rootDir;
		this.datasetDir = new File(rootDir, "training_data/dataset");
		initModel();
	}
	
	/**초기화
	 * @throws Exception
	 */
	protected abstract void initModel() throws Exception;
	
	/**분류 항목의 수가 변경되면 모델 업데이트
	 * @param nOut
	 * @throws Exception
	 */
	protected abstract void updateModel(int nOut) throws Exception;
	
	/**모델 학습 (훈련)
	 * @param datasetDir
	 * @throws Exception
	 */
	public abstract void fit(File datasetDir) throws Exception;
	
	/**모델 테스트
	 * @param features
	 * @return
	 * @throws Exception
	 */
	public abstract PredictResult predict(INDArray features) throws Exception;

	/**모델 테스트
	 * @param imgFile
	 * @return
	 * @throws Exception
	 */
	public PredictResult predict(File imgFile) throws Exception {
		if(modelFile.exists() == false) {
			throw new Exception("아직 학습을 진행하지 않았습니다. 먼저 학습을 진행해 주세요");
		}
		
		log.info("................. NativeImageLoader asMatrix");
		INDArray features = loader.asMatrix(imgFile);
		return predict(features);
	}

	/**모델 테스트
	 * @param img
	 * @return
	 * @throws Exception
	 */
	public PredictResult predict(BufferedImage img) throws Exception {
		if(modelFile.exists() == false) {
			throw new Exception("아직 학습을 진행하지 않았습니다. 먼저 학습을 진행해 주세요");
		}
		log.info("................. NativeImageLoader asMatrix");
		INDArray features = loader.asMatrix(img);
		return predict(features);
	}

	
	/**create DatasetIterator 
	 * @param datasetDir
	 * @return
	 * @throws Exception
	 */
	protected DataSetIterator createDatasetIterator(File datasetDir) throws Exception {

        File[] labelDirs = datasetDir.listFiles(File::isDirectory);
        if (labelDirs == null || labelDirs.length < 2) {
        	throw new IllegalArgumentException("Invalid dataset directory");
        }

        List<DataSet> dataSetList = new ArrayList<>();
        
        for (File labelDir : labelDirs) {
            String labelName = labelDir.getName();
            labelInfo.putLabel(labelName);
            
            File[] imageFiles = labelDir.listFiles(ImageUtils.imgFileFilter());
            if (imageFiles == null || imageFiles.length == 0) {
            	continue;
            }

            for (File imgFile : imageFiles) {
                // Resize and convert to INDArray
                INDArray feature = loader.asMatrix(imgFile);
                preProcessor.transform(feature);

                // One-hot label
                INDArray label = Nd4j.zeros(1, labelDirs.length);
                label.putScalar(0, labelInfo.getLabelIndex(labelName), 1.0);
                dataSetList.add(new DataSet(feature, label));
            }
        }

        // Create and return DataSetIterator
        Collections.shuffle(dataSetList, new Random(123));
        return new ListDataSetIterator<>(dataSetList, DatasetConst.BATCHSIZE.getValue());
    }		
	
}
