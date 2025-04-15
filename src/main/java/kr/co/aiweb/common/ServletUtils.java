package kr.co.aiweb.common;

import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;

public class ServletUtils {

	/**HttpServletRequest
	 * @return
	 */
	public static HttpServletRequest getRequest() {
		return ((ServletRequestAttributes)RequestContextHolder.currentRequestAttributes()).getRequest();
	}
	
	/**HttpServletResponse
	 * @return
	 */
	public static HttpServletResponse getResponse() {
		return ((ServletRequestAttributes)RequestContextHolder.currentRequestAttributes()).getResponse();
	}
	
	/**HttpSession
	 * @return
	 */
	public static HttpSession getSession() {
		return ServletUtils.getRequest().getSession();
	}
	
	/**
	 * session clear
	 */
	public static void clearSession() {
		getSession().invalidate();
	}

}
