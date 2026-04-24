import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

export default defineConfig({
  title: '视觉SLAM学习仪表盘',
  description: '面向《视觉SLAM十四讲》的本地可交互学习文档站',
  lang: 'zh-CN',
  cleanUrls: true,
  lastUpdated: true,

  markdown: {
    lineNumbers: true,
    config: (md) => {
      md.use(mathjax3)
    }
  },

  themeConfig: {
    siteTitle: 'SLAM Learning Dashboard',

    nav: [
      { text: '首页', link: '/' },
      { text: 'C++过渡', link: '/cpp-for-c-programmers/' },
      { text: '核心库', link: '/slam-libraries/' },
      { text: '理论映射', link: '/theory-math-code/' },
      { text: '源码索引', link: '/code-map/' }
    ],

    sidebar: {
      '/cpp-for-c-programmers/': [
        {
          text: '写给 C 程序员的 C++ 指南',
          items: [
            { text: '总览', link: '/cpp-for-c-programmers/' },
            { text: '引用、OOP与多态', link: '/cpp-for-c-programmers/references-oop' },
            { text: '模板与STL', link: '/cpp-for-c-programmers/template-stl' },
            { text: '智能指针与RAII', link: '/cpp-for-c-programmers/smart-pointer-raii' }
          ]
        }
      ],
      '/slam-libraries/': [
        {
          text: 'SLAM核心第三方库',
          items: [
            { text: '总览', link: '/slam-libraries/' },
            { text: 'Eigen', link: '/slam-libraries/eigen' },
            { text: 'Sophus', link: '/slam-libraries/sophus' },
            { text: 'OpenCV', link: '/slam-libraries/opencv' },
            { text: 'Ceres', link: '/slam-libraries/ceres' },
            { text: 'g2o', link: '/slam-libraries/g2o' },
            { text: 'PCL', link: '/slam-libraries/pcl' }
          ]
        }
      ],
      '/theory-math-code/': [
        {
          text: '物理-数学-代码映射',
          items: [
            { text: '总览', link: '/theory-math-code/' },
            { text: '第1章 SLAM全景', link: '/theory-math-code/ch1-intro-and-slam-overview' },
            { text: '第2章 工程与构建', link: '/theory-math-code/ch2-cmake-and-project-basics' },
            { text: '第3章 刚体运动', link: '/theory-math-code/ch3-rigid-body-motion' },
            { text: '第4章 李群李代数', link: '/theory-math-code/ch4-lie-group-and-lie-algebra' },
            { text: '第5章 相机模型', link: '/theory-math-code/ch5-camera-model-and-imaging' },
            { text: '第6章 非线性优化', link: '/theory-math-code/ch6-nonlinear-optimization-basics' },
            { text: '第7章 对极几何', link: '/theory-math-code/ch7-epipolar-geometry-and-triangulation' },
            { text: '第8章 光流与直接法', link: '/theory-math-code/ch8-optical-flow-and-direct-method' },
            { text: '第9章 后端图优化', link: '/theory-math-code/ch9-backend-graph-optimization' },
            { text: '第10章 回环检测', link: '/theory-math-code/ch10-loop-closure-global-consistency' },
            { text: '第11章 建图系统', link: '/theory-math-code/ch11-mapping-system-architecture' },
            { text: '第12章 稠密重建', link: '/theory-math-code/ch12-dense-reconstruction-mapping' },
            { text: '第13章 VIO融合', link: '/theory-math-code/ch13-vio-multi-sensor-fusion' },
            { text: '第14章 工程评估', link: '/theory-math-code/ch14-engineering-evaluation' }
          ]
        }
      ],
      '/code-map/': [
        {
          text: '源码与章节对照',
          items: [
            { text: '总览', link: '/code-map/' },
            { text: '章节映射总表', link: '/code-map/slambook2-chapter-map' }
          ]
        }
      ],
      '/notes/chapters/': [
        {
          text: '章节专题',
          items: [
            { text: '总览', link: '/notes/chapters/' },
            { text: '第9章 后端优化', link: '/notes/chapters/ch9-backend-optimization' },
            { text: '第10章 回环检测', link: '/notes/chapters/ch10-loop-closure' },
            { text: '第11章 建图系统', link: '/notes/chapters/ch11-mapping-system' },
            { text: '第12章 稠密重建', link: '/notes/chapters/ch12-dense-reconstruction' },
            { text: '第13章 VIO融合', link: '/notes/chapters/ch13-vio-fusion' },
            { text: '第14章 工程评估', link: '/notes/chapters/ch14-engineering-practice' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/gaoxiang12/slambook2' }
    ],

    search: {
      provider: 'local'
    },

    outline: {
      level: [2, 3],
      label: '本页目录'
    },

    docFooter: {
      prev: '上一节',
      next: '下一节'
    }
  }
})
