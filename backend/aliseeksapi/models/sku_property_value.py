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


class SkuPropertyValue(object):
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
        'property_value_name': 'str',
        'property_value_id': 'int',
        'property_value_display_name': 'str'
    }

    attribute_map = {
        'property_value_name': 'propertyValueName',
        'property_value_id': 'propertyValueId',
        'property_value_display_name': 'propertyValueDisplayName'
    }

    def __init__(self, property_value_name=None, property_value_id=None, property_value_display_name=None):  # noqa: E501
        """SkuPropertyValue - a model defined in OpenAPI"""  # noqa: E501

        self._property_value_name = None
        self._property_value_id = None
        self._property_value_display_name = None
        self.discriminator = None

        if property_value_name is not None:
            self.property_value_name = property_value_name
        if property_value_id is not None:
            self.property_value_id = property_value_id
        if property_value_display_name is not None:
            self.property_value_display_name = property_value_display_name

    @property
    def property_value_name(self):
        """Gets the property_value_name of this SkuPropertyValue.  # noqa: E501

        The name of a sku value   # noqa: E501

        :return: The property_value_name of this SkuPropertyValue.  # noqa: E501
        :rtype: str
        """
        return self._property_value_name

    @property_value_name.setter
    def property_value_name(self, property_value_name):
        """Sets the property_value_name of this SkuPropertyValue.

        The name of a sku value   # noqa: E501

        :param property_value_name: The property_value_name of this SkuPropertyValue.  # noqa: E501
        :type: str
        """

        self._property_value_name = property_value_name

    @property
    def property_value_id(self):
        """Gets the property_value_id of this SkuPropertyValue.  # noqa: E501

        The ID of the sku value   # noqa: E501

        :return: The property_value_id of this SkuPropertyValue.  # noqa: E501
        :rtype: int
        """
        return self._property_value_id

    @property_value_id.setter
    def property_value_id(self, property_value_id):
        """Sets the property_value_id of this SkuPropertyValue.

        The ID of the sku value   # noqa: E501

        :param property_value_id: The property_value_id of this SkuPropertyValue.  # noqa: E501
        :type: int
        """

        self._property_value_id = property_value_id

    @property
    def property_value_display_name(self):
        """Gets the property_value_display_name of this SkuPropertyValue.  # noqa: E501

        The display name of the sku value   # noqa: E501

        :return: The property_value_display_name of this SkuPropertyValue.  # noqa: E501
        :rtype: str
        """
        return self._property_value_display_name

    @property_value_display_name.setter
    def property_value_display_name(self, property_value_display_name):
        """Sets the property_value_display_name of this SkuPropertyValue.

        The display name of the sku value   # noqa: E501

        :param property_value_display_name: The property_value_display_name of this SkuPropertyValue.  # noqa: E501
        :type: str
        """

        self._property_value_display_name = property_value_display_name

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
        if not isinstance(other, SkuPropertyValue):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
