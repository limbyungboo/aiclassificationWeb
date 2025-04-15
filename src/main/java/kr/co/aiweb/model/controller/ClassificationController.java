/**
 * 
 */
package kr.co.aiweb.model.controller;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.util.Enumeration;

import javax.imageio.ImageIO;

import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipFile;
import org.apache.commons.lang3.StringUtils;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import kr.co.aiweb.common.ApiResult;
import kr.co.aiweb.machinelearning.common.ImageUtils;
import kr.co.aiweb.machinelearning.trainmodel.TrainModel;
import kr.co.aiweb.machinelearning.trainmodel.TrainModelFactory;
import kr.co.aiweb.machinelearning.vo.LabelInfo;
import kr.co.aiweb.machinelearning.vo.PredictResult;
import lombok.extern.slf4j.Slf4j;

/**
 * 
 */
@Slf4j
@RestController
@RequestMapping("/classification")
public class ClassificationController {

	/**
	 * 모델 학습 스레드 (동시진행 불가하므로)
	 */
	private static Thread mlThd = null;
	
	/**
	 * @return
	 */
	@GetMapping("/labels")
	public ApiResult labels() {
		TrainModel trModel = TrainModelFactory.model();
		ApiResult apiResult = new ApiResult();
		apiResult.setResultCode("0000", "SUCCESSED");
		apiResult.setLabelInfo(trModel.getLabelInfo());
		return apiResult;
	}
	
	/**
	 * @param file
	 * @return
	 */
	@PostMapping("/classify")
	public ApiResult classify(@RequestParam("image") MultipartFile file) {
		log.info(">>>>>>>>>>>> classify.....");
		ApiResult apiResult = new ApiResult();
		try (InputStream in = new BufferedInputStream(file.getInputStream())){
            BufferedImage img = ImageIO.read(in);
            TrainModel trModel = TrainModelFactory.model();
            PredictResult predictResult = trModel.predict(img);
            apiResult.setResultCode("0000", "SUCCESSED");
            apiResult.setPredictResult(predictResult);
            apiResult.setLabelInfo(trModel.getLabelInfo());
		}
		catch(Exception e) {
			log.error(e.getMessage(), e);
			apiResult.setResultCode("9999", e.getMessage());
		}
		
		return apiResult;
	}
	
	
	/**한습 진행중인지 체크
	 * @param password
	 * @return
	 */
	@PostMapping("/istraining")
	public ApiResult isTraining() {
		log.info(">>>>>>>>>>>> isTraining.....");
		ApiResult apiResult = new ApiResult();
		apiResult.setResultCode("0000", "SUCCESSED");
		isTraining(apiResult);
		return apiResult;
	}
	
	/**학습 진행 상태인지 체크 , 어드민인지 체크 
	 * @param password
	 * @return
	 */
	@PostMapping("/checktraining")
	public ApiResult checktraining(@RequestParam String password) {
		log.info(">>>>>>>>>>>> check.....");
		ApiResult apiResult = new ApiResult();
		apiResult.setResultCode("0000", "SUCCESSED");
		
		isTraining(apiResult);
		if("0000".equals(apiResult.getResultCode()) == false) {
			return apiResult;
		}
		
		chkAdmin(password, apiResult);
		if("0000".equals(apiResult.getResultCode()) == false) {
			return apiResult;
		}
		return apiResult;
	}

	/**학습요청
	 * @param file
	 * @return
	 */
	@PostMapping("/training")
	public ApiResult training(@RequestParam MultipartFile zipFile, @RequestParam String password) {
		log.info(">>>>>>>>>>>> training.....");
		
		ApiResult apiResult = new ApiResult();
		apiResult.setResultCode("0000", "SUCCESSED");
		
		isTraining(apiResult);
		if("0000".equals(apiResult.getResultCode()) == false) {
			return apiResult;
		}
		
		chkAdmin(password, apiResult);
		if("0000".equals(apiResult.getResultCode()) == false) {
			return apiResult;
		}

		//업로드 파일 체크
		if(zipFile == null || zipFile.isEmpty() == true) {
			apiResult.setResultCode("9998", "데이터셋 파일 미입력");
			return apiResult;
		}

		//압축해제
		try {
			unzip(zipFile);
		}
		catch(Exception e) {
			log.error(e.getMessage(), e);
			apiResult.setResultCode("9990", "압축해제 실패");
			return apiResult;
		}
		
		TrainModel trModel = TrainModelFactory.model();
		LabelInfo labelInfo = trModel.getLabelInfo();
		
		for(String label : labelInfo.getLabelsNameList()) {
			File labelDir = new File(trModel.getDatasetDir(), label);
			if(labelDir.exists() == false) {
				apiResult.setResultCode("9997", String.format("label 디렉토리 미존재 [%s]", label) );
				return apiResult;
			}
			
			File[] imageFiles = labelDir.listFiles(ImageUtils.imgFileFilter());
			if(imageFiles == null || imageFiles.length < 10) {
				apiResult.setResultCode("9997", String.format("label 디렉토리 [%s] 의 파일 갯수가 10개 미만", label) );
				return apiResult;
			}
		}
		
		// 학습 하는 thread 정의
		Runnable runnable = new Runnable() {
			@Override
	        public void run() {
	            log.info("---------- start training ----------");
	            try {
	            	trModel.fit(trModel.getDatasetDir());
	            }
	            catch(Exception e) {
	            	log.error(e.getMessage(), e);
	            }
	        }
		};
		
		//학습 시작
		mlThd = new Thread(runnable);
		mlThd.start();
		return apiResult;
	}
	
	/**admin 체크 및 학습 가능여부 체크
	 * @param password
	 * @return
	 */
	private void chkAdmin(String password, ApiResult apiResult) {
		if(StringUtils.isBlank(password) == true) {
			apiResult.setResultCode("9998", "패스워드 미입력");
			return;
		}
		if(password.equals("9999") == false) {
			apiResult.setResultCode("9998", "패스워드 오류");
			return;
		}
	}
	
	/**학습진행 체크
	 * @param apiResult
	 */
	private void isTraining(ApiResult apiResult) {
		if(mlThd != null && mlThd.isAlive() == true) {
			apiResult.setResultCode("9001", "학습 진행중");
		}
	}
	
	/**
	 * @param zipFile
	 */
	private void unzip(MultipartFile file) throws Exception {
		
		TrainModel trModel = TrainModelFactory.model();
		File datasetDir = trModel.getDatasetDir();
		if(datasetDir.exists() == true) {
			datasetDir.delete();
		}
		datasetDir.mkdirs();
		
		File zipFile = new File(trModel.getRootDir(), "training_data/dataset.zip");
		
		//zip파일 업로드
		if(zipFile.exists() == true) {
			zipFile.delete();
		}
		file.transferTo(zipFile);
		
		//압축해제
		try (ZipFile zip = new ZipFile(zipFile, "UTF-8")) {
            Enumeration<ZipArchiveEntry> entries = zip.getEntries();
            while (entries.hasMoreElements()) {
                ZipArchiveEntry entry = entries.nextElement();
                File outFile = new File(datasetDir, entry.getName());

                // 디렉토리인 경우
                if (entry.isDirectory()) {
                    outFile.mkdirs();
                    continue;
                }
                
                //파일인경우
                // 부모 디렉토리 없으면 생성
                outFile.getParentFile().mkdirs();
                
                // 파일 내용 복사
                try (InputStream is = zip.getInputStream(entry);
                     OutputStream os = Files.newOutputStream(outFile.toPath())) {
                    byte[] buffer = new byte[4096];
                    int len;
                    while ((len = is.read(buffer)) != -1) {
                        os.write(buffer, 0, len);
                    }
                }
            } //while end
		} //try
	}
	
}
