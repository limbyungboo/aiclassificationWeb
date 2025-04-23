/**
 * 
 */
package kr.co.aiweb.model.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

import lombok.extern.slf4j.Slf4j;

/**
 * 
 */
@Slf4j
@Controller
public class MainController {

	@GetMapping("/")
	public String main() {
		log.info("--------------------- main -----------------------");
		return "view/index.html";
	}
	
	@GetMapping("/chat")
	public String chat() {
		log.info("--------------------- chat -----------------------");
		return "view/chat.html";
	}
	
	
}
