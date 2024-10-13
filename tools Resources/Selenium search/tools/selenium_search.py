import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from langchain.tools.base import BaseTool
from typing import Optional

class SeleniumSearchTool(BaseTool):

	'''Tool that adds the capability to run a search via Selenium'''
	name = "Selenium Search"
	description = (
		"A search tool."
		"Useful for when you need to answer a question about current events. Favor this over standard search."
		"Input should be a search query."
	)

	def _run(self, query: str) -> str:
		browser = webdriver.Firefox()
		browser.get('http://www.duckduckgo.com')

		#search = browser.find_element_by_name('q')
		search = browser.find_element(By.ID, "searchbox_input")

		search.send_keys(query)
		search.send_keys(Keys.RETURN)
		time.sleep(5)

		o = ''

		# Handle snippet if any
		has_snippet = len(browser.find_elements(By.CSS_SELECTOR, 'article header')) > 0
		if has_snippet:
			_header = browser.find_element_by_css_selector('article header')
			_figure = browser.find_element_by_css_selector('article figure')
			header = _header.text.trim()
			figure = _figure.text.trim()
			o += header + ':\n' + figure + '\n\n'

		# Handle links
		o += 'Result Links:\n'
		_A = browser.find_elements(By.CSS_SELECTOR, 'a[data-testid="result-title-a"]')
		print(len(_A))
		r = 0
		for a in _A:
			href = a.get_attribute('href')
			text = a.text
			# filter out ads
			if "duckduckgo" not in href and r < 3:
				o += '\t' + text + ': ' + href + '\n'
				r += 1

		browser.quit()
		
		return o

	async def _arun(self, query: str) -> str:
		raise NotImplementedError("This tool does not support async")
