# -*- coding:UTF-8 -*-

import pandas as pd
import numpy as np
import time
from selenium import webdriver
import os
import jieba
import re
import jieba.posseg as pseg
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
    # time.sleep(1)
    # with open('cookies.txt', 'r') as f:
    #     # 使用json读取cookies 注意读取的是文件 所以用load而不是loads
    #     cookies_list = json.load(f)
    #     # 方法1 将expiry类型变为int
    #     for cookie in cookies_list:
    #         # 并不是所有cookie都含有expiry 所以要用dict的get方法来获取
    #         if isinstance(cookie.get('expiry'), float):
    #             cookie['expiry'] = int(cookie['expiry'])
    #         driver.add_cookie(cookie)
    # driver.find_element_by_id("loginname").send_keys("13913865836")
    # driver.find_element_by_class_name("W_input").send_keys("tc19951006")
    #
    # time.sleep(1)
    # driver.find_element_by_class_name("W_btn_a btn_32px").click()

    # driver.close()
def crawlweibobykeywords():
    chromePath = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe'
    driver = webdriver.Chrome(chromePath)
    driver.get('http://weibo.com/login.php')
    time.sleep(20)

    # 进入高级搜索页面
    driver.find_element_by_xpath('//div[@class="gn_header clearfix"]/div[2]/a').click()
    while True:
        try:
            driver.find_element_by_xpath('//div[@class="m-search"]/div[3]/a').click()
            break
        except:
            pass

    wordstr = '四川甘孜州石渠县发生地震'
    province_str = ''  # 陕西
    city_str = ''  # 西安
    s_month = '四月'  # 一月
    s_year = '2020'  # 2019
    s_time = ''  # 8时
    e_month = '四月'  # 一月
    e_year = '2020'  # 2019
    e_time = ''  # 22时

    # 填入关键词
    key_word = driver.find_element_by_xpath('//div[@class="m-layer"]/div[2]/div/div[1]/dl//input')
    key_word.clear()
    key_word.send_keys(wordstr)

    # 填入地点
    try:
        province = driver.find_element_by_xpath('//div[@class="m-adv-search"]/div/dl[5]//select[1]')
        city = driver.find_element_by_xpath('//div[@class="m-adv-search"]/div/dl[5]//select[2]')
        Select(province).select_by_visible_text(province_str)
        Select(city).select_by_visible_text(city_str)
    except:
        pass

    #点击原创
    driver.find_element_by_xpath('//div[@class="m-adv-search"]/div/dl[2]//label[3]/input').click()

    if s_month != '':
        # 填入时间
        # 起始
        driver.find_element_by_xpath('//div[@class="m-adv-search"]/div[1]/dl[4]//input[1]').click()  # 点击input输入框
        sec_1 = driver.find_element_by_xpath('//div[@class="m-caldr"]/div/select[1]')
        Select(sec_1).select_by_visible_text(s_month)
        sec_2 = driver.find_element_by_xpath('//div[@class="m-caldr"]/div/select[2]')
        Select(sec_2).select_by_visible_text(s_year)
        time.sleep(4)  # 输入起始日期
        if s_time != '':
            sec_3 = driver.find_element_by_xpath('//div[@class="m-adv-search"]/div[1]/dl[4]//select[2]')  # 点击input输入框
            Select(sec_3).select_by_visible_text(s_time)  # 小时
    if e_month != '':
        # 终止
        driver.find_element_by_xpath('//div[@class="m-adv-search"]/div[1]/dl[4]//input[2]').click()  # 点击input输入框
        sec_1 = driver.find_element_by_xpath('//div[@class="m-caldr"]/div/select[1]')
        Select(sec_1).select_by_visible_text(e_month)  # 月份
        sec_2 = driver.find_element_by_xpath('//div[@class="m-caldr"]/div/select[2]')
        Select(sec_2).select_by_visible_text(e_year)  # 年份
        time.sleep(4)  # 输入结束日期
        if e_time != '':
            sec_3 = driver.find_element_by_xpath('//div[@class="m-adv-search"]/div[1]/dl[4]//select[2]')  # 点击input输入框
            Select(sec_3).select_by_visible_text(e_time)  # 小时

    time.sleep(1)
    driver.find_element_by_xpath('//div[@class="btn-box"]/a[1]').click()

    # 爬取用户ID 发帖内容 时间  客户端 评论数 转发量 点赞数 并持久化存储
    df = pd.DataFrame({"content": [], "time": [], "博主": [], "博主主页": [], "发布终端": [], "评论数": [], "转发数": [], "点赞数": []})
    while True:
        time.sleep(1)
        div_list = []
        for i in range(4):
            if driver.find_elements_by_xpath('//div[@id="pl_feedlist_index"]/div[2]/div[@action-type="feed_list_item"]'):
                div_list = driver.find_elements_by_xpath('//div[@id="pl_feedlist_index"]/div[2]/div[@action-type="feed_list_item"]')
                break
            else:
                time.sleep(5)
                driver.refresh()
        for div in div_list:
            try:
                user_name = div.find_element_by_xpath('div/div[1]/div[2]/div[1]/div[2]/a[1]').text
                user_url = div.find_element_by_xpath('div/div[1]/div[2]/div[1]/div[2]/a[1]').get_attribute('href')
                try:
                    div.find_element_by_xpath('div/div[1]/div[2]/p[1]/a[@action-type="fl_unfold"]').click()
                    # div.xpath('./div/div[1]/div[2]/p[1]/a[@action-type="fl_unfold"]').click()  # 内容
                    content = div.find_element_by_xpath('div/div[1]/div[2]/p[2]').text
                except:
                    content = div.find_element_by_xpath('div/div[1]/div[2]/p[1]').text  # 内容
                p_time = div.find_element_by_xpath('div/div[1]/div[2]/p[2]/a[1]').text  # 发布时间
                try:
                    client = div.find_element_by_xpath('div/div[1]/div[2]/p[@class="from"]/a[2]').text
                except:
                    client = ''
                up = div.find_element_by_xpath('div/div[2]/ul/li[4]/a').text  # 点赞数
                transfer = re.sub("\D", "", div.find_element_by_xpath('div/div[2]/ul/li[2]/a').text)  # 转发量
                comment = re.sub("\D", "", div.find_element_by_xpath('div/div[2]/ul/li[3]/a').text)  # 评论数
                temp_df = pd.DataFrame({"content": content, "time": p_time, "博主": user_name, "博主主页": user_url, "发布终端": client, "评论数": comment, "转发数": transfer, "点赞数": up}, index=[1])
                df = df.append(temp_df, ignore_index=True)
            except IndexError:
                continue
        try:
            time.sleep(1)
            driver.find_element_by_xpath('//div[@class="m-page"]/div/a[@class="next"]').click()
        except:
            break
    df.to_excel('F:\\weiboEventData\\all\\' + wordstr + '.xlsx')


def crawlweibo():
    chromePath = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe'
    driver = webdriver.Chrome(chromePath)
    driver.get('http://weibo.com/login.php')
    time.sleep(20)
    dirlist = []
    for root, dirs, files in os.walk('F:\\weiboEventData\\all\\'):
        for file in files:
            dirlist.append(os.path.join(root, file))
    for dirl in dirlist:
        df = pd.read_excel(dirl)
        data = np.array(df)
        singleUrl = data[:, 8]
        content = []
        for i in range(len(singleUrl)):
            print(singleUrl[i])
            driver.get(singleUrl[i])
            time.sleep(1)
            try:
                contentdiv = driver.find_element_by_class_name("WB_text").text
                content.append(contentdiv)
                print(content)
            except:
                content.append(" ")
            time.sleep(2)
        df['content'] = content
        df.to_excel(dirl)

def uecdataprocess():
    dirlist = []
    for root, dirs, files in os.walk('F:\\weiboEventData\\true\\'):
        for file in files:
            dirlist.append(os.path.join(root, file))
    for root, dirs, files in os.walk('F:\\weiboEventData\\false\\'):
        for file in files:
            dirlist.append(os.path.join(root, file))
    unexpecteddata = []
    for dirl in dirlist:
        df = pd.read_excel(dirl)
        data = np.array(df['content'])
        unexpecteddata.extend(data)
    all_ue_eventdf = pd.read_excel(r'F:\tc毕业论文实验\ue_eventdata.xls')
    all_ue_eventdf['content'] = unexpecteddata
    all_ue_eventdf['label'] = [1]*len(unexpecteddata)
    all_ue_eventdf.to_excel(r'F:\tc毕业论文实验\ue_eventdata.xls')

def upecdataprocess():
    upec_traindf = pd.read_excel(r'F:\突发事件识别\未处理训练数据.xls')
    upec_traindata = np.array(upec_traindf)
    ec_traindata = []
    for data in upec_traindata:
        if data[9]=='社会生活':
            continue
        elif data[9]=='文体娱乐' or data[9]=='军事' or data[9]=='财经商业' or data[9]=='医药健康' or data[9]=='教育考试' or data[9]=='政治':
            ec_traindata.append(data[1])
        else:
            continue
    ec_traindf = pd.read_excel(r'F:\突发事件识别\实验数据\ec_traindata.xls')
    ec_traindf['content'] = ec_traindata
    ec_traindf.to_excel(r'F:\突发事件识别\实验数据\ec_traindata.xls')

    upec_testdf = pd.read_csv(r'F:\突发事件识别\未处理测试数据.csv')
    upec_testdata = np.array(upec_testdf)
    ec_testdata = []
    for data in upec_testdata:
        if data[9]=='社会生活':
            continue
        elif data[9]=='文体娱乐' or data[9]=='军事' or data[9]=='财经商业' or data[9]=='医药健康' or data[9]=='教育考试' or data[9]=='政治':
            ec_testdata.append(data[1])
        else:
            continue
    ec_testdf = pd.read_excel(r'F:\突发事件识别\实验数据\ec_testdata.xls')
    ec_testdf['content'] = ec_testdata
    ec_testdf.to_excel(r'F:\突发事件识别\实验数据\ec_testdata.xls')


def crawlfalsedata():
    chromePath = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe'
    driver = webdriver.Chrome(chromePath)
    driver.get('http://weibo.com/login.php')
    time.sleep(10)
    driver.get('https://service.account.weibo.com/index?type=5&status=0&page=181')
    df = pd.read_excel('F:\\experiment\\未筛选谣言\\rumor.xlsx')
    page = 181
    while True:
        # for i in range(4):
        #     if driver.find_elements_by_xpath('//div[@id="pl_feedlist_index"]/div[2]/div[@action-type="feed_list_item"]'):
        #         div_list = driver.find_elements_by_xpath('//div[@id="pl_feedlist_index"]/div[2]/div[@action-type="feed_list_item"]')
        #         break
        #     else:
        #         time.sleep(5)
        #         driver.refresh()
        print("页数: "+str(page))
        tr_list = driver.find_elements_by_tag_name('tr')
        for idx in range(len(tr_list)):
            if idx == 0:
                continue
            else:
                print("条数: " + str(idx))
                try:
                    # 点击举报项
                    tr_list[idx].find_element_by_xpath('td[2]/div/a').click()
                    # try:
                    driver.switch_to.window(driver.window_handles[1])
                    time.sleep(1)
                    # 点击原文
                    driver.find_element_by_xpath('//div[@id="pl_service_common"]/div[4]/div[2]/div[1]/div[1]/div[1]/div[1]/p[1]/a').click()
                    driver.switch_to.window(driver.window_handles[2])
                    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, '//div[@class="WB_feed_detail clearfix"]/div[@class="WB_detail"]/div[@class="WB_text W_f14"]')))
                    # except:
                    #     print(2)
                    #     continue
                    try:
                        client = driver.find_element_by_xpath('//div[@class="WB_feed_detail clearfix"]/div[@class="WB_detail"]/div[2]/a[2]').text
                    except:
                        client = ''
                    content = driver.find_element_by_xpath('//div[@class="WB_feed_detail clearfix"]/div[@class="WB_detail"]/div[@class="WB_text W_f14"]').text
                    print(content)
                    p_time = driver.find_element_by_xpath('//div[@class="WB_feed_detail clearfix"]/div[@class="WB_detail"]/div[2]/a[1]').text
                    try:
                        up = re.sub("\D", "", driver.find_element_by_xpath('//div[@class="WB_feed_handle"]/div[1]/ul[1]/li[4]/a[1]/span[1]/span[1]/span[1]/em[2]').text)
                    except:
                        up = 0
                    try:
                        transfer = re.sub("\D", "", driver.find_element_by_xpath('//div[@class="WB_feed_handle"]/div[1]/ul[1]/li[2]/a[1]/span[1]/span[1]/span[1]/em[2]').text)
                    except:
                        transfer = 0
                    try:
                        comment = re.sub("\D", "", driver.find_element_by_xpath('//div[@class="WB_feed_handle"]/div[1]/ul[1]/li[3]/a[1]/span[1]/span[1]/span[1]/em[2]').text)
                    except:
                        comment = 0
                    user_name = driver.find_element_by_xpath('//div[@class="WB_feed_detail clearfix"]/div[@class="WB_detail"]/div[@class="WB_info"]/a[1]').text
                    user_url = driver.find_element_by_xpath('//div[@class="WB_feed_detail clearfix"]/div[@class="WB_detail"]/div[@class="WB_info"]/a[1]').get_attribute('href')
                    url = driver.current_url
                    temp_df = pd.DataFrame(
                        {"content": content, "time": p_time, "博主": user_name, "博主主页": user_url, "发布终端": client,
                         "评论数": comment, "转发数": transfer, "点赞数": up, "url": url}, index=[1])
                    df = df.append(temp_df, ignore_index=True)
                    driver.close()
                    driver.switch_to.window(driver.window_handles[1])
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    time.sleep(1)
                except Exception as e:
                    print(e)
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
        if (page%10) == 0:
            df.to_excel('F:\\experiment\\未筛选谣言\\rumor.xlsx')
            df = pd.read_excel('F:\\experiment\\未筛选谣言\\rumor.xlsx')
        page = page + 1
        try:
            time.sleep(1)
            if driver.find_element_by_xpath('//div[@class="W_pages W_pages_comment"]/a[@class="W_btn_c"][1]').text == '下一页':
                next_url = driver.find_element_by_xpath('//div[@class="W_pages W_pages_comment"]/a[@class="W_btn_c"][1]').get_attribute("href")
                print(next_url)
                driver.get(next_url)
            else:
                next_url = driver.find_element_by_xpath('//div[@class="W_pages W_pages_comment"]/a[@class="W_btn_c"][2]').get_attribute("href")
                driver.get(next_url)
        except:
            break

def crawl_img():
    chromePath = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe'
    driver = webdriver.Chrome(chromePath)
    df = pd.read_excel('F:\\experiment\\未筛选谣言\\rumor2.xlsx')
    img_list = []
    for url in df['url']:
        driver.get(url)


if __name__ == '__main__':

    # chromePath = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe'
    # driver = webdriver.Chrome(chromePath)
    # driver.get('http://weibo.com/login.php')
    # time.sleep(10)
    # driver.get('https://service.account.weibo.com/index?type=5&status=0&page=1')
    # df = pd.DataFrame(
    #     {"content": [], "time": [], "博主": [], "博主主页": [], "发布终端": [], "评论数": [], "转发数": [], "点赞数": [], "url": []})
    # page = 1
    # for t in range(200):
    #     # for i in range(4):
    #     #     if driver.find_elements_by_xpath('//div[@id="pl_feedlist_index"]/div[2]/div[@action-type="feed_list_item"]'):
    #     #         div_list = driver.find_elements_by_xpath('//div[@id="pl_feedlist_index"]/div[2]/div[@action-type="feed_list_item"]')
    #     #         break
    #     #     else:
    #     #         time.sleep(5)
    #     #         driver.refresh()
    #     print("页数: " + str(page))
    #     tr_list = driver.find_elements_by_tag_name('tr')
    #     for idx in range(len(tr_list)):
    #         if idx == 0:
    #             continue
    #         else:
    #             print("条数: " + str(idx))
    #             try:
    #                 # 点击举报项
    #                 tr_list[idx].find_element_by_xpath('td[2]/div/a').click()
    #                 # try:
    #                 driver.switch_to.window(driver.window_handles[1])
    #                 time.sleep(1)
    #                 # 点击原文
    #                 driver.find_element_by_xpath(
    #                     '//div[@id="pl_service_common"]/div[4]/div[2]/div[1]/div[1]/div[1]/div[1]/p[1]/a').click()
    #                 driver.switch_to.window(driver.window_handles[2])
    #                 WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH,
    #                                                                                 '//div[@class="WB_feed_detail clearfix"]/div[@class="WB_detail"]/div[@class="WB_text W_f14"]')))
    #                 # except:
    #                 #     print(2)
    #                 #     continue
    #                 try:
    #                     client = driver.find_element_by_xpath(
    #                         '//div[@class="WB_feed_detail clearfix"]/div[@class="WB_detail"]/div[2]/a[2]').text
    #                 except:
    #                     client = ''
    #                 content = driver.find_element_by_xpath(
    #                     '//div[@class="WB_feed_detail clearfix"]/div[@class="WB_detail"]/div[@class="WB_text W_f14"]').text
    #                 print(content)
    #                 p_time = driver.find_element_by_xpath(
    #                     '//div[@class="WB_feed_detail clearfix"]/div[@class="WB_detail"]/div[2]/a[1]').text
    #                 try:
    #                     up = re.sub("\D", "", driver.find_element_by_xpath(
    #                         '//div[@class="WB_feed_handle"]/div[1]/ul[1]/li[4]/a[1]/span[1]/span[1]/span[1]/em[2]').text)
    #                 except:
    #                     up = 0
    #                 try:
    #                     transfer = re.sub("\D", "", driver.find_element_by_xpath(
    #                         '//div[@class="WB_feed_handle"]/div[1]/ul[1]/li[2]/a[1]/span[1]/span[1]/span[1]/em[2]').text)
    #                 except:
    #                     transfer = 0
    #                 try:
    #                     comment = re.sub("\D", "", driver.find_element_by_xpath(
    #                         '//div[@class="WB_feed_handle"]/div[1]/ul[1]/li[3]/a[1]/span[1]/span[1]/span[1]/em[2]').text)
    #                 except:
    #                     comment = 0
    #                 user_name = driver.find_element_by_xpath(
    #                     '//div[@class="WB_feed_detail clearfix"]/div[@class="WB_detail"]/div[@class="WB_info"]/a[1]').text
    #                 user_url = driver.find_element_by_xpath(
    #                     '//div[@class="WB_feed_detail clearfix"]/div[@class="WB_detail"]/div[@class="WB_info"]/a[1]').get_attribute(
    #                     'href')
    #                 url = driver.current_url
    #                 temp_df = pd.DataFrame(
    #                     {"content": content, "time": p_time, "博主": user_name, "博主主页": user_url, "发布终端": client,
    #                      "评论数": comment, "转发数": transfer, "点赞数": up, "url": url}, index=[1])
    #                 df = df.append(temp_df, ignore_index=True)
    #                 driver.close()
    #                 driver.switch_to.window(driver.window_handles[1])
    #                 driver.close()
    #                 driver.switch_to.window(driver.window_handles[0])
    #                 time.sleep(1)
    #             except Exception as e:
    #                 print(e)
    #                 driver.close()
    #                 driver.switch_to.window(driver.window_handles[0])
    #     if (page % 10) == 0:
    #         df.to_excel('F:\\experiment\\未筛选谣言\\rumor.xlsx')
    #         df = pd.read_excel('F:\\experiment\\未筛选谣言\\rumor.xlsx')
    #     page = page + 1
    #     try:
    #         time.sleep(1)
    #         if driver.find_element_by_xpath(
    #                 '//div[@class="W_pages W_pages_comment"]/a[@class="W_btn_c"][1]').text == '下一页':
    #             next_url = driver.find_element_by_xpath(
    #                 '//div[@class="W_pages W_pages_comment"]/a[@class="W_btn_c"][1]').get_attribute("href")
    #             driver.get(next_url)
    #         else:
    #             next_url = driver.find_element_by_xpath(
    #                 '//div[@class="W_pages W_pages_comment"]/a[@class="W_btn_c"][2]').get_attribute("href")
    #             driver.get(next_url)
    #     except:
    #         break
    # crawlfalsedata()



