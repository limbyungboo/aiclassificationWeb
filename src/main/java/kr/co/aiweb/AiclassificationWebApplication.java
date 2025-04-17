package kr.co.aiweb;

import java.io.File;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.common.config.DL4JSystemProperties;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import kr.co.aiweb.machinelearning.trainmodel.TrainModelFactory;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@SpringBootApplication
public class AiclassificationWebApplication implements CommandLineRunner {

	@Value("${spring.profiles.active}")
	private String active;

	@Value("${machine.learning.model}")
	private String model;

	@Value("${machine.learning.root.dir.window}")
	private String rootdir_window;

	@Value("${machine.learning.root.dir.linux}")
	private String rootdir_linux;
	
	/**main
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			SpringApplication.run(AiclassificationWebApplication.class, args);
		}
		catch(Exception e) {
			log.error(e.getMessage(), e);
			System.exit(1);
		}
	}

	/**@Override 
	 * @see org.springframework.boot.CommandLineRunner#run(java.lang.String[])
	 */
	@Override
	public void run(String... args) throws Exception {
		
		File rootDir;
		
		if("window".equalsIgnoreCase(active) == true) {
			rootDir = new File(rootdir_window);
		}
		else {
			rootDir = new File(rootdir_linux);
		}
		
		if("multilayer".equalsIgnoreCase(model) == false) {
			String baseDir = System.getProperty(DL4JSystemProperties.DL4J_RESOURCES_DIR_PROPERTY);
			log.info(" ######## GET deeplearning4j base dir = {}", baseDir);
			if(StringUtils.isBlank(baseDir) == true) {
				baseDir = String.format("%s/deeplearning4j", rootDir.getAbsolutePath());
				log.info(" ######## SET deeplearning4j base dir = {}", baseDir);
				System.setProperty(DL4JSystemProperties.DL4J_RESOURCES_DIR_PROPERTY, baseDir);
			}
		}
		
		//model 초기화
		TrainModelFactory.initTrainModels(rootDir, model);
	}
}
