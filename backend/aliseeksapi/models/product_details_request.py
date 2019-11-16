# coding: utf-8

"""
    Aliseeks API

    AliExpress product searching and product details retrieval API.   # noqa: E501

    OpenAPI spec version: 1.0.0
    Generated by: https://openapi-generator.tech
"""


import pprint
import re  # noqa: F401

import six


class ProductDetailsRequest(object):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    """
    Attributes:
      openapi_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    openapi_types = {
        'currency': 'RealtimeCurrency',
        'product_id': 'str',
        'locale': 'str',
        'time_zone': 'str',
        'need_store_info': 'bool'
    }

    attribute_map = {
        'currency': 'currency',
        'product_id': 'productId',
        'locale': 'locale',
        'time_zone': 'timeZone',
        'need_store_info': 'needStoreInfo'
    }

    def __init__(self, currency=None, product_id=None, locale='en_US', time_zone='CST', need_store_info=False):  # noqa: E501
        """ProductDetailsRequest - a model defined in OpenAPI"""  # noqa: E501

        self._currency = None
        self._product_id = None
        self._locale = None
        self._time_zone = None
        self._need_store_info = None
        self.discriminator = None

        if currency is not None:
            self.currency = currency
        if product_id is not None:
            self.product_id = product_id
        if locale is not None:
            self.locale = locale
        if time_zone is not None:
            self.time_zone = time_zone
        if need_store_info is not None:
            self.need_store_info = need_store_info

    @property
    def currency(self):
        """Gets the currency of this ProductDetailsRequest.  # noqa: E501


        :return: The currency of this ProductDetailsRequest.  # noqa: E501
        :rtype: RealtimeCurrency
        """
        return self._currency

    @currency.setter
    def currency(self, currency):
        """Sets the currency of this ProductDetailsRequest.


        :param currency: The currency of this ProductDetailsRequest.  # noqa: E501
        :type: RealtimeCurrency
        """

        self._currency = currency

    @property
    def product_id(self):
        """Gets the product_id of this ProductDetailsRequest.  # noqa: E501

        The Product ID   # noqa: E501

        :return: The product_id of this ProductDetailsRequest.  # noqa: E501
        :rtype: str
        """
        return self._product_id

    @product_id.setter
    def product_id(self, product_id):
        """Sets the product_id of this ProductDetailsRequest.

        The Product ID   # noqa: E501

        :param product_id: The product_id of this ProductDetailsRequest.  # noqa: E501
        :type: str
        """

        self._product_id = product_id

    @property
    def locale(self):
        """Gets the locale of this ProductDetailsRequest.  # noqa: E501

        AliExpress locale to use.   # noqa: E501

        :return: The locale of this ProductDetailsRequest.  # noqa: E501
        :rtype: str
        """
        return self._locale

    @locale.setter
    def locale(self, locale):
        """Sets the locale of this ProductDetailsRequest.

        AliExpress locale to use.   # noqa: E501

        :param locale: The locale of this ProductDetailsRequest.  # noqa: E501
        :type: str
        """

        self._locale = locale

    @property
    def time_zone(self):
        """Gets the time_zone of this ProductDetailsRequest.  # noqa: E501

        Timezone to format times in   # noqa: E501

        :return: The time_zone of this ProductDetailsRequest.  # noqa: E501
        :rtype: str
        """
        return self._time_zone

    @time_zone.setter
    def time_zone(self, time_zone):
        """Sets the time_zone of this ProductDetailsRequest.

        Timezone to format times in   # noqa: E501

        :param time_zone: The time_zone of this ProductDetailsRequest.  # noqa: E501
        :type: str
        """

        self._time_zone = time_zone

    @property
    def need_store_info(self):
        """Gets the need_store_info of this ProductDetailsRequest.  # noqa: E501

        Whether we should fetch store information   # noqa: E501

        :return: The need_store_info of this ProductDetailsRequest.  # noqa: E501
        :rtype: bool
        """
        return self._need_store_info

    @need_store_info.setter
    def need_store_info(self, need_store_info):
        """Sets the need_store_info of this ProductDetailsRequest.

        Whether we should fetch store information   # noqa: E501

        :param need_store_info: The need_store_info of this ProductDetailsRequest.  # noqa: E501
        :type: bool
        """

        self._need_store_info = need_store_info

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.openapi_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, ProductDetailsRequest):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other