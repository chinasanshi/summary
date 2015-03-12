///********************************************************************
//created:	2007/11/09
//created:	9:11:2007   15:34
//filename:	CreateXmlFile.cpp
//author:		Wang xuebin
//depend:		libxml2.lib
//build:		nmake TARGET_NAME=CreateXmlFile
//
//purpose:	����һ��xml�ļ�
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
//	//�����ĵ��ͽڵ�ָ��
//	xmlDocPtr doc = xmlNewDoc(BAD_CAST"1.0");//�汾��"1.0"
//	xmlNodePtr root_node = xmlNewNode(NULL, BAD_CAST"root");//�ڶ��������ǽڵ�����
//
//	//���ø��ڵ�
//	xmlDocSetRootElement(doc, root_node);
//
//
//	//�ڸ��ڵ���ֱ�Ӵ����ڵ㣻�ĸ������ֱ�Ϊ�����ڵ㡢����֪���������ڵ����֡��ڵ�����
//	xmlNewTextChild(root_node, NULL, BAD_CAST "newNode1", BAD_CAST "newNode1 content");
//	xmlNewTextChild(root_node, NULL, BAD_CAST "newNode2", BAD_CAST "newNode2 content");
//	xmlNewTextChild(root_node, NULL, BAD_CAST "newNode3", BAD_CAST "newNode3 content");
//
//	//����һ���ڵ㣬���������ݺ����ԣ�Ȼ���������
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
//	//����һ�����Ӻ����ӽڵ�
//	node = xmlNewNode(NULL, BAD_CAST "son");
//	xmlAddChild(root_node, node);
//	xmlNodePtr grandson = xmlNewNode(NULL, BAD_CAST "grandson");
//	xmlAddChild(node, grandson);
//	xmlAddChild(grandson, xmlNewText(BAD_CAST "This is a grandson node"));
//
//	//�洢xml�ĵ�
//	int nRel = xmlSaveFile("CreatedXml2.xml", doc);//���ñ����XML�ļ������֣�nRel����xml�ĵ����ֽ���
//	if (nRel != -1)
//	{
//		cout << "һ��xml�ĵ�������,д��" << nRel << "���ֽ�" << endl;
//	}
//
//	//�ͷ��ĵ��ڽڵ㶯̬������ڴ�
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

purpose:	����xml�ļ�
*********************************************************************/

#include <libxml/parser.h>
#include <iostream>
using namespace std;
int main()
{
	xmlDocPtr doc;			//��������ĵ�ָ�� 
	xmlNodePtr curNode;		//������ָ��(����Ҫ��Ϊ���ڸ��������ƶ�) 
	xmlChar *szKey;			//��ʱ�ַ�������

	char *szDocName;

	szDocName = "CreatedXml2.xml";

	doc = xmlReadFile(szDocName, "GB2312", XML_PARSE_RECOVER);  //�����ļ� 

	//�������ĵ��Ƿ�ɹ���������ɹ���libxml��ָһ��ע��Ĵ���ֹͣ��
	//һ�����������ǲ��ʵ��ı��롣XML��׼�ĵ�������UTF-8��UTF-16�⻹�����������뱣�档
	//����ĵ���������libxml���Զ���Ϊ��ת����UTF-8���������XML������Ϣ������XML��׼��.
	if (NULL == doc)
	{
		fprintf(stderr, "Document not parsed successfully. \n");
		return -1;
	}

	curNode = xmlDocGetRootElement(doc);  //ȷ���ĵ���Ԫ��

	/*���ȷ�ϵ�ǰ�ĵ��а�������*/
	if (NULL == curNode)
	{
		fprintf(stderr, "empty document\n");
		xmlFreeDoc(doc);
		return -1;
	}

	/*����������У�������Ҫȷ���ĵ�����ȷ�����͡���root���������ʾ����ʹ���ĵ��ĸ����͡�*/
	if (xmlStrcmp(curNode->name, BAD_CAST "root"))
	{
		fprintf(stderr, "document of the wrong type, root node != root");
		xmlFreeDoc(doc);
		return -1;
	}

	curNode = curNode->xmlChildrenNode;//�õ��ӽڵ㡣?��ʱcurNodeӦ�������е��ӽڵ�?
	xmlNodePtr propNodePtr = curNode;//?����propNodePtrָ�����ӽڵ���׽ڵ�?
	while (curNode != NULL)
	{
		//ȡ���ڵ��е�����
		if ((!xmlStrcmp(curNode->name, (const xmlChar *)"kyl")))
		{
			szKey = xmlNodeGetContent(curNode);
			printf("newNode1: %s\n", szKey);
			xmlFree(szKey);
		}

		//���Ҵ�������attribute�Ľڵ�
		if (xmlHasProp(curNode, BAD_CAST "attribute"))
		{
			propNodePtr = curNode;
		}
		curNode = curNode->next;
	}

	//��������
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
