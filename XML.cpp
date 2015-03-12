///********************************************************************
//created:	2007/11/09
//created:	9:11:2007   15:34
//filename:	CreateXmlFile.cpp
//author:		Wang xuebin
//depend:		libxml2.lib
//build:		nmake TARGET_NAME=CreateXmlFile
//
//purpose:	创建一个xml文件
//*********************************************************************/
//
//#include <stdio.h>
//#include <libxml/parser.h>
//#include <libxml/tree.h>
//#include <iostream>
//using namespace std;
//
//int main()
//{
//	//定义文档和节点指针
//	xmlDocPtr doc = xmlNewDoc(BAD_CAST"1.0");//版本号"1.0"
//	xmlNodePtr root_node = xmlNewNode(NULL, BAD_CAST"root");//第二个参数是节点名字
//
//	//设置根节点
//	xmlDocSetRootElement(doc, root_node);
//
//
//	//在根节点中直接创建节点；四个参数分别为：根节点、“不知道？”、节点名字、节点内容
//	xmlNewTextChild(root_node, NULL, BAD_CAST "newNode1", BAD_CAST "newNode1 content");
//	xmlNewTextChild(root_node, NULL, BAD_CAST "newNode2", BAD_CAST "newNode2 content");
//	xmlNewTextChild(root_node, NULL, BAD_CAST "newNode3", BAD_CAST "newNode3 content");
//
//	//创建一个节点，设置其内容和属性，然后加入根结点
//	xmlNodePtr node = xmlNewNode(NULL, BAD_CAST"node2");
//	xmlNodePtr content = xmlNewText(BAD_CAST"NODE2 CONTENT");
//	xmlAddChild(root_node, node);
//	xmlAddChild(node, content);
//	xmlNewProp(node, BAD_CAST"attribute", BAD_CAST "yes");
//
//	node = xmlNewNode(NULL, BAD_CAST"kyl");
//	content = xmlNewText(BAD_CAST"bian xie yi ge xml jiedian");
//	xmlAddChild(root_node, node);
//	xmlAddChild(node, content);
//	xmlNewProp(node, BAD_CAST"attribute", BAD_CAST"no");
//
//	//创建一个儿子和孙子节点
//	node = xmlNewNode(NULL, BAD_CAST "son");
//	xmlAddChild(root_node, node);
//	xmlNodePtr grandson = xmlNewNode(NULL, BAD_CAST "grandson");
//	xmlAddChild(node, grandson);
//	xmlAddChild(grandson, xmlNewText(BAD_CAST "This is a grandson node"));
//
//	//存储xml文档
//	int nRel = xmlSaveFile("CreatedXml2.xml", doc);//设置保存的XML文件的名字，nRel保存xml文档的字节数
//	if (nRel != -1)
//	{
//		cout << "一个xml文档被创建,写入" << nRel << "个字节" << endl;
//	}
//
//	//释放文档内节点动态申请的内存
//	xmlFreeDoc(doc);
//	return 0;
//}


/********************************************************************
created:	2007/11/15
created:	15:11:2007   11:47
filename:	ParseXmlFile.cpp
author:		Wang xuebin
depend:		libxml2.lib
build:		nmake TARGET_NAME=ParseXmlFile

purpose:	解析xml文件
*********************************************************************/

#include <libxml/parser.h>
#include <iostream>
using namespace std;
int main()
{
	xmlDocPtr doc;			//定义解析文档指针 
	xmlNodePtr curNode;		//定义结点指针(你需要它为了在各个结点间移动) 
	xmlChar *szKey;			//临时字符串变量

	char *szDocName;

	szDocName = "CreatedXml2.xml";

	doc = xmlReadFile(szDocName, "GB2312", XML_PARSE_RECOVER);  //解析文件 

	//检查解析文档是否成功，如果不成功，libxml将指一个注册的错误并停止。
	//一个常见错误是不适当的编码。XML标准文档除了用UTF-8或UTF-16外还可用其它编码保存。
	//如果文档是这样，libxml将自动地为你转换到UTF-8。更多关于XML编码信息包含在XML标准中.
	if (NULL == doc)
	{
		fprintf(stderr, "Document not parsed successfully. \n");
		return -1;
	}

	curNode = xmlDocGetRootElement(doc);  //确定文档根元素

	/*检查确认当前文档中包含内容*/
	if (NULL == curNode)
	{
		fprintf(stderr, "empty document\n");
		xmlFreeDoc(doc);
		return -1;
	}

	/*在这个例子中，我们需要确认文档是正确的类型。“root”是在这个示例中使用文档的根类型。*/
	if (xmlStrcmp(curNode->name, BAD_CAST "root"))
	{
		fprintf(stderr, "document of the wrong type, root node != root");
		xmlFreeDoc(doc);
		return -1;
	}

	curNode = curNode->xmlChildrenNode;//得到子节点。?此时curNode应该是所有的子节点?
	xmlNodePtr propNodePtr = curNode;//?定义propNodePtr指向了子节点的首节点?
	while (curNode != NULL)
	{
		//取出节点中的内容
		if ((!xmlStrcmp(curNode->name, (const xmlChar *)"kyl")))
		{
			szKey = xmlNodeGetContent(curNode);
			printf("newNode1: %s\n", szKey);
			xmlFree(szKey);
		}

		//查找带有属性attribute的节点
		if (xmlHasProp(curNode, BAD_CAST "attribute"))
		{
			propNodePtr = curNode;
		}
		curNode = curNode->next;
	}

	//查找属性
	xmlAttrPtr attrPtr = propNodePtr->properties;
	while (attrPtr != NULL)
	{
		if (!xmlStrcmp(attrPtr->name, BAD_CAST "attribute"))
		{
			xmlChar* szAttr = xmlGetProp(propNodePtr, BAD_CAST "attribute");
			cout << "get attribute = " << szAttr << endl;
			xmlFree(szAttr);
		}
		attrPtr = attrPtr->next;
	}

	xmlFreeDoc(doc);
	return 0;
}
